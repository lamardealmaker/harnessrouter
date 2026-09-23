#!/usr/bin/env bash
# Sourced by the entrypoint and by native-binary CI. Expects TOOLS to name the install root.
# Unreal Agent, MIT. Install its native binary and preserve the upstream licence.
# The archive digests are pinned to v0.1.1, not fetched from a mutable latest release.
UNREAL_PIN="0.1.1"
install_unreal() {
  local ur_arch ur_sha ur_tmp ur_have
  case "$(uname -m)" in
    x86_64) ur_arch="amd64"; ur_sha="fad9cb9e6e6272a8d16fb4b90f985abb3132572413588f96622c6b1a82e34fcd" ;;
    aarch64|arm64) ur_arch="arm64"; ur_sha="0e61571dc9b83b429aaf9c89d8af372ff39a7ef50313fa2fd0ef15c6fa527d02" ;;
    *) echo "unsupported architecture $(uname -m) for Unreal Agent"; return 1 ;;
  esac
  ur_tmp="$(mktemp -d)"
  curl -fsSL --retry 2 --connect-timeout 15 --max-time 180 \
    "https://github.com/unreallabsai/unreal-agent/releases/download/v${UNREAL_PIN}/unreal-agent-runner_${UNREAL_PIN}_linux_${ur_arch}.tar.gz" \
    -o "$ur_tmp/unreal.tar.gz" || { rm -rf "$ur_tmp"; return 1; }
  ur_have="$(sha256sum "$ur_tmp/unreal.tar.gz" | awk '{print $1}')"
  if [ "$ur_sha" != "$ur_have" ]; then
    echo "Unreal Agent $UNREAL_PIN: checksum mismatch for $ur_arch"
    rm -rf "$ur_tmp"; return 1
  fi
  tar --no-same-owner -xzf "$ur_tmp/unreal.tar.gz" -C "$ur_tmp" unreal-agent-runner LICENSE \
    || { rm -rf "$ur_tmp"; return 1; }
  "$ur_tmp/unreal-agent-runner" -h >/dev/null 2>&1 || { rm -rf "$ur_tmp"; return 1; }
  mkdir -p "$TOOLS/bin" "$TOOLS/unreal" && \
    install -m 755 "$ur_tmp/unreal-agent-runner" "$TOOLS/bin/unreal-agent-runner" && \
    install -m 644 "$ur_tmp/LICENSE" "$TOOLS/unreal/LICENSE" && \
    printf '%s\n' "$UNREAL_PIN" > "$TOOLS/unreal/version" \
    || { rm -rf "$ur_tmp"; return 1; }
  rm -rf "$ur_tmp"
}

