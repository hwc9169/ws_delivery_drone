import argparse
import asyncio
import os
import sys


async def _spawn(env: dict, *args: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(*args, env=env)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Launch two drone agents and delivery orchestration.")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--agent-config", default="src/delivery/agent/config.yaml")
    parser.add_argument("--orchestration-config", default="src/delivery/orchestration/config.yaml")
    parser.add_argument("--drone1-address", default="udpin://:14540")
    parser.add_argument("--drone2-address", default="udpin://:14541")
    args = parser.parse_args()

    python = sys.executable or "python3"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src" + (f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")
    processes = [
        await _spawn(
            env,
            python,
            "-u",
            "-m",
            "delivery.agent.drone_agent",
            "--drone-id",
            "drone1",
            "--system-address",
            args.drone1_address,
            "--mqtt-host",
            args.mqtt_host,
            "--mqtt-port",
            str(args.mqtt_port),
            "--config",
            args.agent_config,
        ),
        await _spawn(
            env,
            python,
            "-u",
            "-m",
            "delivery.agent.drone_agent",
            "--drone-id",
            "drone2",
            "--system-address",
            args.drone2_address,
            "--mqtt-host",
            args.mqtt_host,
            "--mqtt-port",
            str(args.mqtt_port),
            "--config",
            args.agent_config,
        ),
        await _spawn(
            env,
            python,
            "-u",
            "-m",
            "delivery.orchestration.delivery_orchestration",
            "--mqtt-host",
            args.mqtt_host,
            "--mqtt-port",
            str(args.mqtt_port),
            "--config",
            args.orchestration_config,
        ),
    ]

    try:
        await asyncio.gather(*(p.wait() for p in processes))
    except asyncio.CancelledError:
        raise
    finally:
        for proc in processes:
            if proc.returncode is None:
                proc.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
