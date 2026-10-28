"""Parser mínimo de datagramas sFlow v5 (amostras de cabeçalho de pacote).

Extrai, de cada amostra de fluxo com registro "raw packet header", os campos
necessários aos atributos: IP de origem/destino, protocolo, flags TCP e portas.
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

ETH_IPV4 = 0x0800
ETH_VLAN = 0x8100
TCP_SYN, TCP_ACK = 0x02, 0x10


@dataclass
class PacketSample:
    src: str
    dst: str
    proto: int
    sport: int = 0
    dport: int = 0
    tcp_flags: int = 0
    sampling_rate: int = 1

    @property
    def is_syn(self) -> bool:
        return self.proto == 6 and (self.tcp_flags & TCP_SYN) and not (self.tcp_flags & TCP_ACK)


def _u32(b, o):
    return struct.unpack_from("!I", b, o)[0]


def parse_header(hdr: bytes, rate: int) -> PacketSample | None:
    if len(hdr) < 14:
        return None
    etype = struct.unpack_from("!H", hdr, 12)[0]
    off = 14
    if etype == ETH_VLAN and len(hdr) >= 18:
        etype = struct.unpack_from("!H", hdr, 16)[0]
        off = 18
    if etype != ETH_IPV4 or len(hdr) < off + 20:
        return None
    ihl = (hdr[off] & 0x0F) * 4
    proto = hdr[off + 9]
    src = socket.inet_ntoa(hdr[off + 12:off + 16])
    dst = socket.inet_ntoa(hdr[off + 16:off + 20])
    s = PacketSample(src, dst, proto, sampling_rate=rate)
    l4 = off + ihl
    if proto in (6, 17) and len(hdr) >= l4 + 4:
        s.sport, s.dport = struct.unpack_from("!HH", hdr, l4)
        if proto == 6 and len(hdr) >= l4 + 14:
            s.tcp_flags = hdr[l4 + 13]
    return s


def parse_datagram(data: bytes) -> list[PacketSample]:
    """Devolve as amostras de pacote contidas em um datagrama sFlow v5."""
    out: list[PacketSample] = []
    try:
        if _u32(data, 0) != 5:
            return out
        addr_type = _u32(data, 4)
        o = 8 + (4 if addr_type == 1 else 16)
        o += 12                                   # sub_agent_id, seq, uptime
        n_samples = _u32(data, o)
        o += 4
        for _ in range(n_samples):
            fmt, length = _u32(data, o), _u32(data, o + 4)
            body = o + 8
            o = body + length
            enterprise, kind = fmt >> 12, fmt & 0xFFF
            if enterprise != 0 or kind not in (1, 3):
                continue                          # amostras de contador etc.
            if kind == 1:                         # flow_sample
                rate = _u32(data, body + 8)
                p = body + 32
            else:                                 # expanded flow_sample
                rate = _u32(data, body + 12)
                p = body + 44
            n_rec = _u32(data, p - 4)
            for _ in range(n_rec):
                rfmt, rlen = _u32(data, p), _u32(data, p + 4)
                rbody = p + 8
                p = rbody + rlen
                if rfmt != 1:                     # raw packet header
                    continue
                hlen = _u32(data, rbody + 12)
                hdr = data[rbody + 16: rbody + 16 + hlen]
                s = parse_header(hdr, rate)
                if s:
                    out.append(s)
    except (struct.error, IndexError, OSError):
        pass
    return out


def build_datagram(headers: list[bytes], rate: int = 64) -> bytes:
    """Monta um datagrama sFlow v5 (usado nos testes unitários)."""
    samples = b""
    for i, h in enumerate(headers):
        pad = (4 - len(h) % 4) % 4
        rec = struct.pack("!IIII", 1, len(h) + 4, 0, len(h)) + h + b"\0" * pad
        rec = struct.pack("!II", 1, len(rec)) + rec
        body = struct.pack("!IIIIIIII", i, 1, rate, i * rate, 0, 1, 2, 1) + rec
        samples += struct.pack("!II", 1, len(body)) + body
    head = struct.pack("!II", 5, 1) + socket.inet_aton("127.0.0.1") + struct.pack("!IIII", 0, 1, 0, len(headers))
    return head + samples
