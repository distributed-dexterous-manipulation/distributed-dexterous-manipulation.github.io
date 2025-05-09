""" These are local IP addresses of the individual delta array modules. Each module controls 4 robots. 
    Will need to be changed for different setups. """

delta_comm_dict = {
    "192.168.1.29": 16,
    "192.168.1.30": 15,
    "192.168.1.31": 8,
    # "192.168.1.32": 11,
    "192.168.1.32": 11,
    "192.168.1.33": 3,
    "192.168.1.34": 4,
    "192.168.1.35": 7,
    "192.168.1.36": 12,
    "192.168.1.7": 9,
    "192.168.1.38": 6,
    "192.168.1.39": 10,
    "192.168.1.40": 1,
    "192.168.1.41": 14,
    "192.168.1.42": 5,
    "192.168.1.43": 13,
    "192.168.1.44": 2,
}

inv_delta_comm_dict = {v: k for k, v in delta_comm_dict.items()}
# print(inv_delta_comm_dict)