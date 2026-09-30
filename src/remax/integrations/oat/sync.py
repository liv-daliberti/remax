"""OAT sync responsibilities, preserving historical state and ordering."""

from __future__ import annotations
import logging
import socket
import time
import torch
import torch.distributed as dist
from oat.utils.distributed import (
    init_process_group,
    node_ip_address_from_perspective,
    torch_type_codec,
)


class OatSyncMixin:
    def _init_local_actor_weight_sync(self) -> None:
        """Replace rank-zero fanout with four concurrent learner/actor pairs."""

        if not bool(getattr(self.args, "local_actor_weight_sync", False)):
            return
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if int(self.args.num_gpus_per_actor) != 1:
            raise RuntimeError("local actor sync requires one GPU per actor")
        if len(self.actors) != world_size:
            raise RuntimeError(
                "local actor sync requires exactly one actor per learner rank"
            )
        master_addr = node_ip_address_from_perspective()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        group_name = f"oat_local_actor_sync_{rank}"
        actor = self.actors[rank]
        actor_future = actor.futures.init_process_group(
            master_addr,
            master_port,
            1,
            2,
            group_name,
            "gloo",
        )
        local_group = init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=2,
            rank=0,
            group_name=group_name,
        )
        actor_future.result()
        self._local_sync_actor = actor
        self._local_model_update_group = local_group
        dist.barrier()
        logging.info(
            "local actor weight-sync group initialized learner_rank=%s actor=%s",
            rank,
            rank,
        )

    def sync_params_to_actors(self):
        if not bool(getattr(self.args, "local_actor_weight_sync", False)):
            return super().sync_params_to_actors()

        started = time.time()
        dist.barrier()
        actor = self._local_sync_actor
        reset_future = (
            actor.futures.reset_prefix_cache()
            if self.args.enable_prefix_caching
            else None
        )
        model = self.model.model.module
        parameters = list(model.named_parameters())
        torch.cuda.empty_cache()
        for index, (name, param) in enumerate(parameters, start=1):
            actor_future = actor.futures.update_weight(
                name,
                dtype=torch_type_codec(param.dtype),
                shape=param.shape,
                empty_cache=index == len(parameters),
            )
            dist.broadcast(
                param.data,
                0,
                group=self._local_model_update_group,
            )
            actor_future.result()
        if reset_future is not None:
            reset_future.result()
        if int(getattr(self.args, "vllm_sleep_level", 1)) == 2:
            actor.wake_up(["kv_cache"])
        torch.cuda.empty_cache()
        dist.barrier()
        self.pi_beta_version += 1
        self.pi_beta_lags_behind = False
        self.weight_sync_elapse = time.time() - started
        logging.info(
            "weights @version=%s broadcasted through local actor pair in %.3fs",
            self.pi_beta_version,
            self.weight_sync_elapse,
        )

    def _pre_learning(self):
        if int(getattr(self.args, "vllm_sleep_level", 1)) != 2:
            return super()._pre_learning()
        started = time.time()
        torch.cuda.synchronize()
        dist.barrier()
        if self.strategy.is_group_rank_0():
            backup_futures = [
                actor.futures.backup_model_buffers() for actor in self.actors
            ]
            backup_reports = [future.result() for future in backup_futures]
            logging.info("backed up vLLM model buffers: %s", backup_reports)
            futures = [actor.futures.sleep(2) for actor in self.actors]
            _ = [future.result() for future in futures]
        torch.cuda.synchronize()
        dist.barrier()
        self.vllm_go_sleep_time = time.time() - started
        logging.info(
            "vLLM actors entered discard-mode sleep in %.3fs",
            self.vllm_go_sleep_time,
        )

    def _post_learning(self):
        if int(getattr(self.args, "vllm_sleep_level", 1)) != 2:
            return super()._post_learning()
        started = time.time()
        torch.cuda.synchronize()
        dist.barrier()
        if self.strategy.is_group_rank_0():
            futures = [actor.futures.wake_up(["weights"]) for actor in self.actors]
            _ = [future.result() for future in futures]
            restore_futures = [
                actor.futures.restore_model_buffers() for actor in self.actors
            ]
            restore_reports = [future.result() for future in restore_futures]
            logging.info("restored vLLM model buffers: %s", restore_reports)
        torch.cuda.synchronize()
        dist.barrier()
        self.vllm_wake_up_time = time.time() - started
        self.pi_beta_lags_behind = True
        logging.info(
            "vLLM actor weight storage remapped in %.3fs",
            self.vllm_wake_up_time,
        )
