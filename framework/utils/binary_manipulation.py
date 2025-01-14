import struct

def bin2float(binary_list):
    rez=''.join(binary_list)
    h = int(rez, 2).to_bytes(8, byteorder="big")
    return struct.unpack('>d', h)[0]

# 64bit representation
def float2bin(float_val):
    [d] = struct.unpack(">Q", struct.pack(">d", float_val))
    return list(f'{d:064b}')


def solution2binary(solution_x):
    binary_sol=[]
    for val in solution_x:
        a=float2bin(val)
        binary_sol+=a
    return binary_sol

def binarysolution2float(binary_sol):
    n=64
    solution_x=[]
    bitss=[binary_sol[i:i+n] for i in range(0, len(binary_sol), n)]
    for bits in bitss:
        solution_x.append(bin2float(bits))
    return solution_x