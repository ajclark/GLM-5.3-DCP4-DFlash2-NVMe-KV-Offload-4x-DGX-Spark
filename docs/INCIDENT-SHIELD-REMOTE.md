# A TV remote powered off two DGX Sparks: the NVIDIA SHIELD Remote incident (2026-09-06)

On the evening of 2026-09-05 (Pacific) two of the four DGX Sparks in this
cluster shut themselves down cleanly, 97 seconds apart, while nobody was in
the room. The cause was a Bluetooth NVIDIA SHIELD Remote that had once been
paired with those two machines. Its power button, pressed to turn on a TV,
was delivered to the Sparks as a keyboard power key, and Linux did what it
does with a power key by default: an orderly shutdown. This note records the
evidence, the mechanism, the second-order failure it caused, and the fixes.

## 1. What happened

| time (PDT, 09-05) | event |
|---|---|
| 17:40:21 | spark-a218: `systemd-logind: Power key pressed short.` → `Powering off...` |
| 17:40:22-24 | spark-ddbf: `kernel: input: NVIDIA SHIELD Remote as .../uhid/0005:0955:7217.0001/input/input7`; `hid-generic ... BLUETOOTH HID v0.02 Gamepad [NVIDIA SHIELD Remote]`; `systemd-logind: Watching system buttons on /dev/input/event7 (NVIDIA SHIELD Remote)` |
| 17:40:33 | spark-a218 reaches `poweroff.target`; its ring neighbour spark-06c4 logs `enP2p1s0f0np0: Link down` a second later |
| 17:41:58 | spark-ddbf: `systemd-logind: Power key pressed short.` → `Powering off...` |
| 17:42:11 | spark-ddbf reaches `poweroff.target`; spark-365c logs `Link down` on its port to ddbf |
| ~17:45-18:45 | cluster wall power ~100 W (two idle Sparks); UniFi shows both 10GbE ports offline; no ARP from either node |
| 18:39 | a chat request to the serving stack hangs: rank 0's engine waits for workers that no longer exist (`/health` still returns 200) |
| 18:46 | both nodes powered back on by hand; previous-boot journals read |

Both journals show a textbook orderly shutdown (filesystems synced, services
stopped, `last -x` records "shutdown system down"). No kernel error, nothing
in pstore, no kdump. The survivors had nothing in their logs but the
neighbours' links dropping.

The remote had been connected to spark-a218 since that node's boot on
2026-08-31 (`Watching system buttons on /dev/input/event7 (NVIDIA SHIELD
Remote)` at 12:32:23 that day). When a218 shut down, the remote lost its
link and, being bonded and trusted on spark-ddbf as well, reconnected there
within a second. A second press 94 seconds later took ddbf down. The two
nodes without a paired remote never noticed.

`bluetoothctl` on the two nodes today:

```
Device 48:B0:2D:39:56:1F NVIDIA SHIELD Remote
        Paired: yes   Bonded: yes   Trusted: yes   Connected: no
```

spark-06c4 and spark-365c have no paired Bluetooth devices.

## 2. Why a remote's power button shuts a server down

Three defaults line up:

1. **The remote's power key is a plain HID power key.** The SHIELD Remote
   (USB ID 0955:7217) enumerates over Bluetooth as a HID device; the kernel
   binds `hid-generic` and creates an input device whose keymap includes
   `KEY_POWER`. The kernel calls it a "Gamepad", which does not matter.
2. **systemd tags every key-capable input device as a power switch.** The
   shipped udev rule `70-power-switch.rules` is two lines:
   `SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_SWITCH}=="1", TAG+="power-switch"`
   and the same for `ENV{ID_INPUT_KEY}=="1"`. Any keyboard-like device gets
   the tag; `logind.conf(5)` says "Only input devices with the power-switch
   udev tag will be watched for key/lid switch events", and logind logs
   `Watching system buttons on /dev/input/eventN (<name>)` for each.
3. **logind's default for a short power-key press is poweroff.**
   `HandlePowerKey=` defaults to `poweroff`; `HandlePowerKeyLongPress=`
   defaults to `ignore`. There is no per-device policy: the chassis button
   and a remote in another room are the same event.

So a Bluetooth remote, keyboard, or game controller with a power key that
was ever paired and trusted can power the machine off from wherever it is in
range, with no session, no login and no confirmation. On a headless server
that boots to `multi-user.target` nothing ever displays a "powering off"
screen either.

Reports of the same pattern with other devices are easy to find: servers
shut down by "Power key pressed" events from USB or wireless keyboards and
their control endpoints ([Arch forums](https://bbs.archlinux.org/viewtopic.php?id=264918),
[Oracle Linux](https://community.oracle.com/customerconnect/discussion/637440/oracle-linux-server-got-shutdown-with-systemd-logind-power-key-pressed-message)),
and users discovering `HandlePowerKey` while trying to stop it
([EndeavourOS](https://forum.endeavouros.com/t/handlepowerkey-setting-for-logind-not-working/55539),
[Fedora](https://discussion.fedoraproject.org/t/how-do-i-disable-the-power-button/89663),
[openSUSE](https://forums.opensuse.org/t/systemd-logind-understanding-power-button-options/170808);
background in [Baeldung](https://www.baeldung.com/linux/power-button-behavior)).

## 3. Why it was easy to blame the wrong thing

Seventy minutes earlier the cluster had been through a ConnectX-7 hot-plug
power experiment (`docs/CX7-POWER.md`), and the two nodes that died were
exactly the two on the newer kernel and driver build. Both facts were
coincidence. What settled it was reading the previous boot's journal on the
nodes themselves after power-on (`journalctl -b -1`), which showed
`Power key pressed short` rather than a crash, and then asking which input
devices logind had been watching. The first draft of the incident record
said "someone pressed the power button"; the cluster's owner, who knew
nobody had been near the machines, pushed back, and the Bluetooth trail
proved them right. An input-event log line is evidence of an input event,
not of a hand.

## 4. The second-order failure: GID index 3

Bringing the stack back exposed a latent fragility. Both launchers pin
`NCCL_IB_GID_INDEX=3`, which assumes each ring port's IPv4 RoCEv2 GID sits at
index 3 of the adapter's GID table. On the two survivors, the ring port
whose peer had rebooted came back with that GID at **index 4** and index 3
empty (the kernel re-added the IP's GIDs into different slots after the link
flap). NCCL on every rank then failed with

```
misc/ibvwrap.cc:380 (wrap_ibv_modify_qp) NCCL WARN Call to ibv_modify_qp failed with 61
No data available, on dev roceP2p1s0f0:1, curr state INIT, next state RTR, local GID index 3, local GID ::
```

and both the DCP launch and the production fallback exited within 75 s.
Bouncing the NetworkManager connection on the affected port
(`nmcli con down roce-p0; nmcli con up roce-p0`) re-packs the table to
index 3. `rollout_dcp.sh` now checks both ring ports on every node before it
stops anything and performs that bounce itself.

Two more things do not survive a node reboot and are now handled: the
2000 MHz GPU clock lock (a `gpu-clock-lock.service` oneshot is enabled on
all four nodes) and the NVMe slab cache (wiped on boot-id change by design).

## 5. Fixes, cheapest first

**Applied 2026-09-06 02:10 UTC at the owner's request:** the remote was unpaired on
both nodes (`bluetoothctl remove 48:B0:2D:39:56:1F`; no paired devices remain
anywhere) and Bluetooth is soft-blocked on all four nodes (`rfkill block
bluetooth`, state persisted by `systemd-rfkill`, adapter reports Powered: no).
The logind change below was not applied; it remains available.

- **Unpair the remote** on the two nodes:
  `bluetoothctl remove 48:B0:2D:39:56:1F`. The remote is otherwise useless
  to a headless server.
- **Turn Bluetooth off on cluster nodes.** `rfkill block bluetooth` (soft,
  persists across reboots via `systemd-rfkill`), or disable Wi-Fi/Bluetooth
  in UEFI, which the January 2026 DGX OS release notes added for exactly
  this class of environment.
- **Make the power key harmless in software.** In
  `/etc/systemd/logind.conf.d/10-power-key.conf`:
  `[Login]` / `HandlePowerKey=ignore` / `HandlePowerKeyLongPress=poweroff`,
  then `systemctl restart systemd-logind`. A short press from any device is
  ignored; a deliberate long press on the chassis button still powers off.
  Power-on still needs the button in any case (Spark has no Wake-on-LAN).
- **Targeted alternative** if the remote must stay paired for some reason:
  a udev rule that strips the tag from it,
  `SUBSYSTEM=="input", ATTRS{name}=="NVIDIA SHIELD Remote", TAG-="power-switch"`.
- **Detect it next time.** `journalctl -u systemd-logind | grep 'Watching
  system buttons'` lists every device that can power the box off;
  `bluetoothctl devices Paired` lists what can reconnect on its own. Both
  belong in a post-boot check for a headless cluster.

## 6. Takeaways for anyone running Sparks (or any Linux box) headless

- A DGX Spark ships with Bluetooth on and a full BlueZ stack. Pairing a
  remote or keyboard once is enough for it to power the machine off later,
  from another room, with no interaction on the host.
- The failure looks like a hardware fault from the outside (nodes vanish
  from the network, wall power drops) and like a clean shutdown from the
  inside. Read the previous boot's journal before theorising.
- Multi-node inference does not degrade when a rank disappears: the API
  keeps answering `/health` while the engine waits forever. Probe with a
  real generation.
- Anything you pin to a device index (GID index, interface names, PCI
  addresses) should be checked at launch, because a neighbour's reboot can
  move it.

Evidence files: `results/incident-20260906-two-nodes-down/` (both nodes'
previous-boot captures, the timeline, the capture script).
