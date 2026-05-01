# Sandbox VM — Setup Guide

Krakey routes every non-idempotent / privacy-touching operation
(coding, CLI, file I/O, GUI control, browser) through a guest VM so
the host is insulated from anything she does. This is **default-on**;
the runtime will refuse to start if you enable a sandboxed tool
without configuring the VM.

**Scope of Phase S1 (this release):** `coding` tool only. Other
sandboxed tools land in S2–S3.

---

## Architecture

```
┌─ HOST (Krakey runtime, dashboard, GM/KB) ──────────────┐
│                                                        │
│   CodingTool ──► SandboxRunner ─── HTTP/RPC ─┐     │
│                                                  │     │
│   ┌──────────────── GUEST VM ───────────────┐   │     │
│   │  krakey-agent (HTTP server)             │◄──┘     │
│   │                                         │         │
│   │   POST /exec   — run cmd, return        │         │
│   │   GET  /health — version + status       │         │
│   └─────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────┘
```

**Network contract.**
- VM NIC 1: NAT → internet, open. Krakey can curl / pip / apt freely.
- VM NIC 2: host-only bridge. Agent listens here only.
- Guest firewall: DROP RFC1918 destinations except the host-only
  subnet. **VM cannot reach your LAN or host services**.
- Host → VM: only over the agent URL (host-initiated RPC).
- VM → host: **never**. Agent responds but does not initiate.

---

## 1. Provision the VM

Recommended: **QEMU** (cross-platform, scriptable). On Linux it uses
KVM; on macOS HVF; on Windows WHPX or run under WSL2.

Pick whatever guest OS you prefer:

| Guest | Why |
|---|---|
| Linux (Debian / Ubuntu server) | lightest, cheapest, best for coding / CLI |
| macOS | only if host is macOS (legal + HVF limit) |
| Windows | if you need to test Windows-only software |

Minimum resources: 2 vCPU, 4 GB RAM, 40 GB disk.

**Display mode — headed vs headless (pick one):**

- **headed** — Krakey's VM desktop shows in a window on your screen. You
  can see what she is doing, intervene manually, log in from inside.
  Launch QEMU with `-display sdl` (or `spice` / `gtk`). Cost: a
  visible window + a display server running inside the guest.
- **headless** — VM runs with no display. Cheaper resources, less
  noise, but you cannot watch or help interactively. Launch QEMU
  with `-display none` (or `-nographic`). Coding-only sandboxes
  generally want this; GUI-tool sandboxes need headed.

The `sandbox.display` config field is **declarative only** — the
runtime does not start the VM for you. Set it to the mode you
intended; Phase S4 lifecycle tooling will consume it.

### Example: Ubuntu Server 22.04 under QEMU on Linux host

```bash
# One-time: allocate disk
qemu-img create -f qcow2 krakey-sandbox.qcow2 40G

# Install from ISO (first boot only — headed window so you can click through)
qemu-system-x86_64 \
  -enable-kvm -cpu host -smp 2 -m 4096 \
  -drive file=krakey-sandbox.qcow2,if=virtio \
  -cdrom ubuntu-22.04-live-server-amd64.iso \
  -boot d \
  -netdev user,id=net0 -device virtio-net,netdev=net0 \
  -display sdl
```

Inside the VM installer: create a normal user (e.g. `krakey`), enable
sudo, install openssh-server. No GUI needed for coding-only sandbox.

After install, shut down and relaunch **without the ISO** and with a
second host-only NIC for the agent:

```bash
qemu-system-x86_64 \
  -enable-kvm -cpu host -smp 2 -m 4096 \
  -drive file=krakey-sandbox.qcow2,if=virtio \
  -netdev user,id=net0,ipv6=off -device virtio-net,netdev=net0 \
  -netdev user,id=ag,net=10.0.3.0/24,hostfwd=tcp:127.0.0.1:18765-:8765 \
  -device virtio-net,netdev=ag \
  -display sdl
```

The `hostfwd` clause forwards host's `127.0.0.1:18765` into the VM's
port 8765 — the agent's HTTP endpoint.

---

## 2. Install the Guest Agent

Inside the VM, copy `krakey/environment/sandbox/agent.py` from this
repo onto the guest (any path — e.g. `/opt/krakey/agent.py`) and run
it under systemd so it restarts if the VM reboots.

```bash
# On the guest:
sudo mkdir -p /opt/krakey
sudo cp agent.py /opt/krakey/agent.py
sudo chmod +x /opt/krakey/agent.py

# Generate a shared token (keep it matching host config)
openssl rand -hex 32 > /opt/krakey/token
```

Create `/etc/systemd/system/krakey-agent.service`:

```ini
[Unit]
Description=Krakey Guest Agent
After=network-online.target

[Service]
Type=simple
User=krakey
Environment=SANDBOX_AGENT_TOKEN=<paste the token you generated>
ExecStart=/usr/bin/python3 /opt/krakey/agent.py \
    --host 0.0.0.0 --port 8765 \
    --workspace /home/krakey/work
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now krakey-agent
sudo systemctl status krakey-agent
```

Verify from the host:

```bash
curl -H "X-Krakey-Token: <token>" http://127.0.0.1:18765/health
# → {"status":"ok","guest_os":"linux","agent_version":"1",...}
```

---

## 3. Lock Down Guest Networking

Inside the guest, block outbound to the host's LAN so Krakey cannot
scan your network:

```bash
# Find the RFC1918 ranges you want to block (keep the agent subnet open)
sudo iptables -I OUTPUT -d 10.0.0.0/8     -j DROP
sudo iptables -I OUTPUT -d 172.16.0.0/12  -j DROP
sudo iptables -I OUTPUT -d 192.168.0.0/16 -j DROP
# Allow the host-only net (agent) back in
sudo iptables -I INPUT  -s 10.0.3.0/24    -j ACCEPT
sudo iptables -I OUTPUT -d 10.0.3.0/24    -j ACCEPT

# Persist with iptables-persistent or your distro's equivalent.
sudo apt install -y iptables-persistent
sudo netfilter-persistent save
```

Leave public internet (NAT NIC) unrestricted per the network policy
you confirmed.

---

## 4. Configure Krakey

Edit `config.yaml`:

```yaml
environments:
  local:
    # Plugins allow-listed for direct host execution. Leave empty if
    # every code-running plugin should go through the sandbox.
    allowed_plugins: []
  sandbox:
    # Plugins allow-listed for VM-isolated execution.
    allowed_plugins: [coding]
    guest_os: linux                     # or macos / windows
    provider: qemu
    vm_name: krakey-sandbox
    display: headed                     # headed | headless (your choice)
    resources:
      cpu: 2
      memory_mb: 4096
      disk_gb: 40
    agent:
      url: "http://127.0.0.1:18765"     # matches hostfwd above
      token: "${SANDBOX_AGENT_TOKEN}"   # env-substituted
    network_mode: nat_allowlist         # documentation; enforced in guest fw
    allowlist_domains: []               # not yet enforced
```

The plugin gets the env via `ctx.environment("sandbox").run(...)`;
the Router refuses if the plugin is not on the `allowed_plugins`
list above.

Export the token on the host:

```bash
export SANDBOX_AGENT_TOKEN="<same token you put in the guest service>"
```

Start Krakey. At boot you'll see:

```
[HB #0] sandbox preflight ok: guest_os=linux agent_version=1
```

If the agent is unreachable or the token mismatches, the runtime
**refuses to start** with an explicit error — fix the config rather
than disabling the sandbox.

---

## 5. Snapshots

Phase S1 does not manage snapshots automatically. Use QEMU's native
support:

```bash
# Freeze current state as a named snapshot
qemu-img snapshot -c baseline krakey-sandbox.qcow2
qemu-img snapshot -c day-7    krakey-sandbox.qcow2

# List
qemu-img snapshot -l krakey-sandbox.qcow2

# Roll back (VM must be off)
qemu-img snapshot -a baseline krakey-sandbox.qcow2
```

Snapshots cover everything **inside** the VM (installed packages,
downloaded files, Krakey's temporary work). They do **not** touch
Krakey's GM/KB/self_model — those live on the host and persist
independently.

Recommended ritual:
- Take a `baseline` snapshot immediately after provisioning.
- Snapshot before any risky experiment.
- Never give Krakey the ability to roll back; that is yours alone.

---

## Opting Out (Unsafe)

If you want to test without a VM, assign the plugin to the `local`
env instead of `sandbox`:

```yaml
environments:
  local:
    allowed_plugins: [coding]    # runs in host process — UNSAFE
  # Omit the `sandbox:` block entirely; nothing else needs changing.
```

**This runs the subprocess directly on your host with your user's
privileges.** Use only on throwaway machines.
