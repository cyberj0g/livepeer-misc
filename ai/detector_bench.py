"""
This script creates a CSV report from livepeer_bench output by running it with all combinations of number of sessions and detection frequency parameters

Example (all paths are relative to livepeer_bench executable path, and results file will be created in livepeer_bench dir):

python3 detector_bench.py --livepeer-bench ../../go-livepeer/livepeer_bench --max-sessions 2 --in-file ../bbb/source.m3u8
"""

import os
import re
import subprocess
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import argparse
import tqdm
import nvidia_smi
from itertools import product

pd.options.display.width = 0
pd.set_option('display.max_columns', None)

LPB_CMD_LINE = './livepeer_bench  -in {in_file} -transcodingOptions transcodingOptions.json -concurrentSessions {sessions} -detectionFreq {detection_freq} -nvidia {gpu_num} -outPrefix /tmp/'

def split_no_empty(string, sep=' '):
    return list(filter(None, string.split(sep)))

def read_human(metric_str):
    if 'g' in metric_str:
        metric = float(metric_str.replace('g', ''))*1024**3
    elif 'm' in metric_str:
        metric = float(metric_str.replace('g', ''))*1024**2
    else:
        metric = float(metric_str)
    return metric

def get_process_stats(pid):
    top_proc = subprocess.Popen(("top -b -n 1 -d 0.1 -p %s" % pid).split(), stdout=subprocess.PIPE)
    cpu_mem_str = split_no_empty(top_proc.communicate()[0].decode('ascii'), '\n')[-1]
    cpu = read_human(split_no_empty(cpu_mem_str)[8])
    # top output is in kb
    mem = read_human(split_no_empty(cpu_mem_str)[5])*1024
    # read GPU stats
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return cpu, mem, res.gpu, mem_res.used

def read_stream_lines(stream):
    line = stream.readline().decode('utf-8').strip()
    if line is None or line=='':
        return []
    return [line]

def capture_process_data(arguments):
    res = defaultdict(lambda: [])
    proc = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res['returncode'] = None
    res['start_time'] = time.time()
    stderr_finish = False
    stdout_finish = False
    while proc.returncode is None:
        time.sleep(0.1)
        while True:
            # get cpu and ram usage
            cpu, mem, gpu, gpu_mem = get_process_stats(proc.pid)
            res['cpu_hist'].append(cpu)
            res['ram_hist'].append(mem)
            res['gpu_hist'].append(gpu)
            res['vram_hist'].append(gpu_mem)
            lines = read_stream_lines(proc.stdout)
            err_lines = read_stream_lines(proc.stderr)
            if not err_lines and not lines:
                break
            else:
                res['lines'].extend(lines)
                res['lines'].extend(err_lines)
        proc.poll()
        if proc.returncode is not None:
            res['time_sec'] = round(time.time() - res['start_time'], 2)
    res['returncode'] = proc.returncode
    try:
        res['tail'] = '\n' + '\t\n'.join(res['lines'][-3:]) + '\n'
    except:
        pass
    return res


def create_cmd_line(args):
    cmd_line = LPB_CMD_LINE
    for arg, val in args.items():
        arg_key = '{' + arg + '}'
        cmd_line = cmd_line.replace(arg_key, str(val))
    return cmd_line


def run_bench(args):
    sessions = list(range(1, args.max_sessions+1))
    #sessions = [1]
    #freqs = list(range(0, args.max_detection_freq+1))
    freqs = [0, 1, 2, 5, 10, 30]
    #freqs = [0]
    args_grid = product(sessions, freqs)
    res_list = []
    for sess, freq in args_grid:
        runtime_args = {'sessions': sess, 'detection_freq': freq}
        runtime_args.update(vars(args))
        cmd_line = create_cmd_line(runtime_args)
        print(cmd_line)
        res = capture_process_data(cmd_line.split())
        if res['returncode'] == 0:
            for l in res['lines']:
                if 'Real-Time Segs Ratio' in l:
                    res['rt_seg_ratio'] = float(split_no_empty(l, '|')[-1])
                if 'Total Transcoding Duration' in l:
                    res['total_t_time'] = float(split_no_empty(l, '|')[-1].replace('s', ''))
                if 'Real-Time Duration Ratio' in l:
                    res['rt_dur_ratio'] = float(split_no_empty(l, '|')[-1])
        res['max_cpu'] = np.max(res['cpu_hist'])
        res['max_gpu'] = np.max(res['gpu_hist'])
        res['max_ram'] = np.max(res['ram_hist'])
        res['max_vram'] = np.max(res['vram_hist'])

        res['avg_cpu'] = np.mean(res['cpu_hist'])
        res['avg_gpu'] = np.mean(res['gpu_hist'])
        res['avg_ram'] = np.mean(res['ram_hist'])
        res['avg_vram'] = np.mean(res['vram_hist'])

        res['sess_count'] = sess
        res['detect_freq'] = freq
        res_list.append(res)
        df = pd.DataFrame(res_list)
        df.drop(['lines', 'cpu_hist', 'gpu_hist', 'ram_hist', 'vram_hist', 'tail'], axis=1, inplace=True)
        df.to_csv('bench_results.csv')
        print(df)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--livepeer-bench', default='../../go-livepeer/livepeer_bench')
    ap.add_argument('--in-file', default='../../data/bbb/source.m3u8')
    ap.add_argument('--max-sessions', default=10, type=int)
    ap.add_argument('--max-detection-freq', default=10, type=int)
    ap.add_argument('--gpu-num', default=0)
    args = ap.parse_args()
    os.chdir(os.path.dirname(args.livepeer_bench))
    nvidia_smi.nvmlInit()
    run_bench(args)
