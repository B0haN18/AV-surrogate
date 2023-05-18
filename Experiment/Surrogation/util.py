#! /usr/bin/python3


def print_debug(info):
    print(info)
    with open('output.log', 'a') as f:
        print(info, file=f)
        f.close()

def print_res(info):
    print(info)
    with open('res.log', 'a') as f:
        print(info, file=f)
        f.close()
