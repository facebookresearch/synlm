# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import getpass
import itertools
import os
import pickle
import random
import re
import socket
import subprocess
import sys
from logging import getLogger
from math import ceil

import torch


from .logger import create_logger


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

DUMP_PATH = "/checkpoint/%s/dumped"
DUMP_PATH = DUMP_PATH % getpass.getuser()


logger = getLogger()

def download_snippet(snippet):
    local_path = f"/checkpoint/asablayrolles/synlm/{snippet}/checkpoint.pt"

    if not os.path.exists(local_path):
        return 0

    return local_path


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = "SLURM_JOB_ID" in os.environ and not params.debug_slurm
    logger.info("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:

        assert params.local_rank == -1  # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NTASKS",
            "SLURM_TASKS_PER_NODE",
            "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU",
            "SLURM_NODEID",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_TASK_PID",
        ]

        PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            logger.info(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ["SLURM_JOB_ID"]

        # number of nodes / node ID
        params.n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        params.node_id = int(os.environ["SLURM_NODEID"])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ["SLURM_LOCALID"])
        params.global_rank = int(os.environ["SLURM_PROCID"])

        # number of processes / GPUs per node
        params.world_size = int(os.environ["SLURM_NTASKS"])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        params.master_addr = hostnames.split()[0].decode("utf-8")
        assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        logger.info(PREFIX + "Master address: %s" % params.master_addr)
        logger.info(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ["MASTER_ADDR"] = params.master_addr
        os.environ["MASTER_PORT"] = str(params.master_port)
        os.environ["WORLD_SIZE"] = str(params.world_size)
        os.environ["RANK"] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["NGPU"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "Global rank    : %i" % params.global_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        logger.info("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger, log_stream = create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=getattr(params, "global_rank", 0),
        snippet=params.snippet,
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join(
            f"{k}: {str(v)} ({type(v)})" for k, v in sorted(dict(vars(params)).items())
        )
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")

    return logger, log_stream


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    if params.exp_name == "bypass":
        dump_path = params.dump_path.rstrip("/")
        params.exp_id = os.path.basename(dump_path)
        sweep_path = os.path.dirname(dump_path)
        if sweep_path == "":
            sweep_path = "/tmp"
    else:
        dump_path = DUMP_PATH if params.dump_path == "" else params.dump_path
        assert len(params.exp_name) > 0
        assert os.path.isdir(dump_path)
        # create the sweep path if it does not exist
        sweep_path = os.path.join(dump_path, params.exp_name)
        if not os.path.exists(sweep_path):
            subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # randomly generated
    if params.exp_id == "":
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_oom_error(exception: BaseException) -> bool:
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_out_of_cpu_memory(exception)
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
# def garbage_collection_cuda() -> None:
#     """Garbage collection Torch (CUDA) memory."""
#     gc.collect()
#     try:
#         # This is the last thing that should cause an OOM error, but seemingly it can.
#         torch.cuda.empty_cache()
#     except RuntimeError as exception:
#         if not is_oom_error(exception):
#             # Only handle OOM errors
#             raise
def garbage_collection_cuda():
    torch.cuda.empty_cache()
