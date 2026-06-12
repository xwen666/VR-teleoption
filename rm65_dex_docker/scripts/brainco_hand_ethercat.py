#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Optional


def can_import_brainco_sdk() -> bool:
    try:
        from bc_stark_sdk import main_mod as sdk  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class BraincoEthercatDeviceInfo:
    description: str = ""
    slave_pos: int = 0
    master_pos: int = 0


class BraincoEthercatHandClient:
    def __init__(
        self,
        master_pos: int = 0,
        slave_pos: int = 0,
        cycle_ns: int = 1_000_000,
        command_duration_ms: int = 20,
    ):
        self.master_pos = int(master_pos)
        self.slave_pos = int(slave_pos)
        self.cycle_ns = int(cycle_ns)
        self.command_duration_ms = max(1, int(command_duration_ms))

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._stopped = threading.Event()
        self._connect_error: Optional[BaseException] = None
        self._sdk = None
        self._ctx = None
        self.device_info = BraincoEthercatDeviceInfo(
            slave_pos=self.slave_pos,
            master_pos=self.master_pos,
        )

    def _expected_device_paths(self) -> list[str]:
        return [
            f"/dev/EtherCAT{self.master_pos}",
            f"/dev/ethercat{self.master_pos}",
            "/dev/EtherCAT0",
            "/dev/ethercat0",
        ]

    def _format_preflight_hint(self) -> str:
        expected = ", ".join(self._expected_device_paths())
        return (
            "BrainCo EtherCAT backend could not find the local EtherCAT master device. "
            f"Expected one of: {expected}. "
            "This usually means the IgH EtherCAT master is not installed, not started, "
            "or the NIC has not been bound to the EtherCAT driver yet."
        )

    def connect(self, timeout_sec: float = 8.0) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        if not self._connected.wait(timeout_sec):
            raise TimeoutError("Timed out while connecting to the BrainCo EtherCAT hand backend.")
        if self._connect_error is not None:
            cause = self._connect_error
            extra = ""
            message = str(cause)
            if "No such file or directory" in message:
                extra = " " + self._format_preflight_hint()
            raise RuntimeError(
                "Failed to initialize BrainCo EtherCAT hand backend. "
                "Make sure bc-stark-sdk and the IgH EtherCAT master are installed, "
                "and that the Revo2 hand is visible on the selected master/slave position. "
                f"Underlying error: {type(cause).__name__}: {message}.{extra}"
            ) from cause

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._async_connect())
        except BaseException as exc:  # pragma: no cover - startup path
            self._connect_error = exc
            self._connected.set()
        else:
            self._connected.set()
            try:
                self.loop.run_forever()
            finally:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            self._stopped.set()
            self.loop.close()

    async def _async_connect(self) -> None:
        from bc_stark_sdk import main_mod as sdk

        self._sdk = sdk
        self._ctx = sdk.init_device_handler(sdk.StarkProtocolType.EtherCAT, self.master_pos)
        await self._ctx.ec_setup_sdo(self.slave_pos)
        info = await self._ctx.get_device_info(self.slave_pos)
        self.device_info = BraincoEthercatDeviceInfo(
            description=getattr(info, "description", str(info)),
            slave_pos=self.slave_pos,
            master_pos=self.master_pos,
        )
        await self._ctx.set_finger_unit_mode(self.slave_pos, sdk.FingerUnitMode.Normalized)
        await self._ctx.ec_reserve_master()
        await self._ctx.ec_start_loop([self.slave_pos], 0, self.cycle_ns, 0, 0, 0)

    def _submit(self, coro, timeout_sec: float = 2.0):
        if self.loop is None or self._ctx is None:
            raise RuntimeError("BrainCo EtherCAT hand backend is not connected.")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout_sec)

    async def _async_set_positions(self, positions: list[int], duration_ms: int) -> int:
        await self._ctx.set_finger_positions_and_durations(
            self.slave_pos,
            [int(value) for value in positions],
            [int(duration_ms)] * 6,
        )
        return 0

    def set_positions(self, positions: list[int], duration_ms: Optional[int] = None) -> int:
        clipped = [max(0, min(1000, int(value))) for value in positions[:6]]
        duration_ms = self.command_duration_ms if duration_ms is None else max(1, int(duration_ms))
        return int(self._submit(self._async_set_positions(clipped, duration_ms)))

    async def _async_get_positions(self) -> list[int]:
        status = await self._ctx.get_motor_status(self.slave_pos)
        return [int(value) for value in list(status.positions)[:6]]

    def get_positions(self) -> list[int]:
        return self._submit(self._async_get_positions())

    async def _async_close(self) -> None:
        if self._ctx is None:
            return
        try:
            await self._ctx.ec_stop_loop()
        finally:
            await self._ctx.close()

    def close(self, timeout_sec: float = 3.0) -> None:
        if self.loop is None:
            return
        if not self._connected.is_set() or self._ctx is None:
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass
            if self.thread is not None:
                self.thread.join(timeout=timeout_sec)
            self.thread = None
            self.loop = None
            self._ctx = None
            self._sdk = None
            return
        try:
            self._submit(self._async_close(), timeout_sec=timeout_sec)
        except Exception:
            pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        if self.thread is not None:
            self.thread.join(timeout=timeout_sec)
        self.thread = None
        self.loop = None
        self._ctx = None
        self._sdk = None
