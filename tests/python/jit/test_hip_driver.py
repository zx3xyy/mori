from __future__ import annotations

from mori.jit import hip_driver


def test_get_hip_lib_reuses_loaded_runtime_before_rocm_path(monkeypatch):
    calls = []

    class FakeCDLL:
        def __init__(self, path, mode=None):
            calls.append((path, mode))

    monkeypatch.setattr(hip_driver.ctypes, "CDLL", FakeCDLL)
    monkeypatch.setattr(hip_driver, "_hip", None)

    hip_driver._get_hip_lib()

    assert calls[0][0] == "libamdhip64.so"
    assert calls[0][1] & hip_driver.os.RTLD_NOLOAD


def test_get_hip_lib_uses_rocm_path_when_no_runtime_is_loaded(monkeypatch, tmp_path):
    rocm_path = tmp_path / "rocm"
    expected = str(rocm_path / "lib" / "libamdhip64.so")
    calls = []

    class FakeCDLL:
        def __init__(self, path, mode=None):
            calls.append((path, mode))
            if mode is not None and mode & hip_driver.os.RTLD_NOLOAD:
                raise OSError("not found")

    monkeypatch.setenv("ROCM_PATH", str(rocm_path))
    monkeypatch.setattr(hip_driver.ctypes, "CDLL", FakeCDLL)
    monkeypatch.setattr(hip_driver, "_hip", None)

    hip_driver._get_hip_lib()

    assert calls[-1] == (expected, None)


def test_launch_struct_uses_kernel_params(monkeypatch):
    class FakeHip:
        def __init__(self):
            self.calls = []

        def hipModuleLaunchKernel(self, *args):
            self.calls.append(args)
            return 0

    fake_hip = FakeHip()

    monkeypatch.setattr(hip_driver, "_get_hip_lib", lambda: fake_hip)

    func = hip_driver.HipFunction(hip_driver.c_void_p(0x1234), "kernel")
    func.launch_struct(
        grid=(2, 3),
        block=(4,),
        shared_mem=5,
        stream=6,
        struct_ptr=0x7890,
    )

    assert len(fake_hip.calls) == 1
    launch_args = fake_hip.calls[0]
    assert launch_args[9]
    assert launch_args[10].value is None


def test_launch_multi_uses_kernel_params(monkeypatch):
    class FakeHip:
        def __init__(self):
            self.calls = []

        def hipModuleLaunchKernel(self, *args):
            self.calls.append(args)
            return 0

    fake_hip = FakeHip()
    monkeypatch.setattr(hip_driver, "_get_hip_lib", lambda: fake_hip)

    hip_driver.launch_multi(
        funcs=[hip_driver.c_void_p(0x1234), hip_driver.c_void_p(0x5678)],
        grids=[2, 3],
        blocks=[4, 5],
        shared_mems=[6, 7],
        stream=8,
        struct_ptr=0x7890,
    )

    assert len(fake_hip.calls) == 2
    for launch_args in fake_hip.calls:
        assert launch_args[9]
        assert launch_args[10].value is None
