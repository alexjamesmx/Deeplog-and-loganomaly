import argparse
from logging import Logger
import pandas as pd


def sliding_window(df, window_size, step_size, logger: Logger):
    """
    Generate fixed/sliding window for the dataset
    if step_size == window_size, then it is fixed window 
    if step_size < window_size, then it is sliding window 
    Args:
        df (_type_): _description_
        window_size (_type_): _description_
        step_size (_type_): _description_
        logger (Logger): _description_

    Returns:
        _type_: _description_
    """
    log_size = df.shape[0]
    timestamp, severity, events, message, log_uuid,  = df["_zl_timestamp"], df[
        "SEVERITY"], df["EVENTID"],  df["MESSAGE"], df["log_uuid"]

    print(f"first 10 before process {events[:1]}")

    # severity_values = df["SEVERITY"].unique()
    # print(f"Severity values: {severity_values}")
    # severity_mapping = {severity: i for i,
    # severity in enumerate(severity_values)}
    # print(f"Severity mapping: {severity_mapping}")
    # Apply the mapping to create a new numerical column
    # severity = df['SEVERITY'].map(severity_mapping)
    print(f"Severity: {severity}")

    new_data = []
    start_end_index_pair = []

    start_index = 0
    while start_index < log_size:
        end_index = min(start_index + window_size, log_size)
        start_end_index_pair.append(tuple([start_index, end_index]))
        start_index = start_index + step_size

    n_sess = 0
    # print(f"Start end index pair: {start_end_index_pair[:10]}")
    for (start_index, end_index) in start_end_index_pair:
        new_data.append({
            "SessionId": n_sess,
            "_zl_timestamp": timestamp[start_index:end_index].values.tolist(),
            "SEVERITY": severity[start_index:end_index].values.tolist(),
            "EventId": events[start_index: end_index].values.tolist(),
            "MESSAGE": message[start_index: end_index].values.tolist(),
            "log_uuid": log_uuid[start_index: end_index].values.tolist(),
        })
        n_sess += 1

    assert len(start_end_index_pair) == len(new_data)

    # print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
    logger.info(f"Number of sessions: {len(new_data)}")
    # logger.info(f"session one {new_data[0]}")
    # print("proccess new ", new_data[:3])
    return new_data

# def time_window(df, window_size: 60, step_size: 60,
#                 logger: Logger):
#     # df = df[:1000]
#     # print(df)
#     window_size = window_size * 1000
#     step_size = step_size * 1000

#     log_size = df.shape[0]
#     time_data, severity = df["_zl_timestamp"], df["SEVERITY"]
#     events, messages = df["EVENTID"], df["MESSAGE"]
#     print(f"Log size: {log_size}")
#     new_data = []
#     start_end_index_pair = set()

#     # get the first start time, start index, end index, end time
#     start_time = int(time_data[0])
#     end_time = start_time + window_size
#     start_index = 0
#     end_index = 0

#     for cur_time in time_data:
#         if int(cur_time) < end_time:
#             end_index += 1
#         else:
#             break

#     start_end_index_pair.add(tuple([start_index, end_index]))

#     # print(f"Start time: {start_time}, end time: {end_time}")
#     # print(f"start_end_index_pair: {start_end_index_pair}")
#     while end_index < log_size:
#         start_time = start_time + step_size
#         end_time = start_time + window_size
#         for i in range(start_index, log_size):
#             if int(time_data[i]) < start_time:
#                 i += 1
#             else:
#                 break
#         for j in range(end_index, log_size):
#             if int(time_data[j]) < end_time:
#                 j += 1
#             else:
#                 break
#         start_index = i
#         end_index = j

#         # when start_index == end_index, there is no value in the window
#         if start_index != end_index:
#             start_end_index_pair.add(tuple([start_index, end_index]))

#     print(f"Total Number of windows: {len(start_end_index_pair)}")
#     n_sess = 0
#     for (start_index, end_index) in start_end_index_pair:
#         # print(f"Start index: {start_index}, end index: {end_index}")
#         new_data.append({
#             "Label": severity[start_index:end_index].values.tolist(),
#             "EventId": events[start_index: end_index].values.tolist(),
#             "messages": messages[start_index: end_index].values.tolist(),
#             "SessionId": n_sess,
#         })
#         n_sess += 1

#     assert len(start_end_index_pair) == len(new_data)
#     print(f"Number of sessions: {len(new_data)}")
#     # print(new_data)
#     # print(f"{new_data[-1]}")
#     return new_data
