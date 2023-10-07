from phe.paillier import PaillierPrivateKey, PaillierPublicKey, generate_paillier_keypair
import numpy as np
import warnings
import multiprocessing
from joblib import Parallel, delayed
from numba import njit, prange
import binary_float_decimal
import torch
import copy
import warnings

N_JOBS = multiprocessing.cpu_count()
public_key, private_key = generate_paillier_keypair(n_length=2048)
def encrypt(public_key: PaillierPublicKey, x):
    return public_key.encrypt(x)

def encrypt_array(public_key: PaillierPublicKey, A):
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    return np.array(encrypt_A)

def encrypt_matrix(public_key: PaillierPublicKey, A: np.array):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)
    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    encrypt_A = np.expand_dims(encrypt_A, axis=0)
    encrypt_A = np.reshape(encrypt_A, og_shape)
    return np.array(encrypt_A)

# @njit(parallel=True)
def add_threshold(input, threshold_dict: dict):  # input is  clients_weight_after_train : dict
    for client_idx in range(len(input)):
        for k in input[client_idx].keys():
            input[client_idx][k] += threshold_dict[k]
    return input

def de_threshold(input, threshold: int, num_clients: int):  # input :ndarray
    return input - threshold * num_clients

# @njit(parallel=True)
def f2b_matrix(input, M=8, K=6, N=30):
    result = Parallel(n_jobs=N_JOBS)(delayed(binary_float_decimal.f2b)(i, M, K, N) for i in input)
    return np.array(result)


def splicing(B: np.array) -> str:
    return ''.join(B)


def encrypt_matrix_batch(public_key: PaillierPublicKey, A, batch_size=4, M=8, K=6, N=30):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)
    A = np.reshape(A, (1, -1)) 
    A = np.squeeze(A)
    A_len = len(A)
    # pad array at the end so tha the array is the size of
    A = A if (A_len % batch_size) == 0 \
        else np.pad(A, (0, batch_size - (A_len % batch_size)), 'constant', constant_values=(0, 0))
    A = f2b_matrix(A, M, K, N)
    idx_range = int(len(A) / batch_size)
    batched_nums = []

    new_arr = np.array_split(A, idx_range)  
    for i in range(idx_range):
        batched_one = splicing(new_arr[i])
        batched_nums.append(batched_one)
    batched_nums = np.array(batched_nums)  
    encoded_A = Parallel(n_jobs=N_JOBS)(delayed(int)(num, 2) for num in batched_nums) 
    encrypted_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in encoded_A)
    return encrypted_A, og_shape


def decrypt(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)

def restore_shape(decrypt_A, shape, batch_size, M=8, K=6, N=30):
    batch_bits = (M + N) * batch_size
    decoded_A = Parallel(n_jobs=N_JOBS)(delayed(binary_float_decimal.decode_fillzero)(num, batch_bits) for num in decrypt_A)  
    num_ele = np.prod(shape)
    num_ele_w_pad = batch_size * len(decoded_A)
    un_batched_nums = []
    for t in range(len(decoded_A)):
        move = 0
        for j in range(batch_size):
            tail = move + M + N
            un_batched_nums.append(decoded_A[t][move:tail])
            move = tail
    un_batched_nums = np.array(un_batched_nums)
    un_batched_nums_2_str = Parallel(n_jobs=N_JOBS)(
        delayed(binary_float_decimal.b2f)(i, M, K, N) for i in un_batched_nums)
    un_batched_nums_2_str = np.array(un_batched_nums_2_str).astype(np.float64)
    res = np.reshape(un_batched_nums_2_str[0:num_ele], shape)
    return res


def decrypt_matrix_batch(private_key: PaillierPrivateKey, A, og_shape, batch_size=4, M=8, K=6, N=30):
    decrypt_A = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(num) for num in A) 
    decrypt_A = np.array(decrypt_A)
    result = restore_shape(decrypt_A, og_shape, batch_size, M, K, N)
    return result


def batch_encrypt_per_layer(publickey: PaillierPrivateKey, party: dict, batch_size: int, M: dict, K: int, N: dict):
    result: dict = {}
    og_shapes: dict = {}  
    for k in party.keys():
        enc, shape_ = encrypt_matrix_batch(publickey, party[k].cpu().numpy(), batch_size=batch_size, M=M[k],
                                           K=K, N=N[k])
        result[k] = enc
        og_shapes[k] = shape_
    return result, og_shapes


def batch_decrypt_per_layer(privatekey: PaillierPrivateKey, party: dict, og_shap: dict, batch_size: int, M: dict,
                            K: int, N: dict):
    result = {}
    for k in party.keys():
        result[k] = decrypt_matrix_batch(private_key=privatekey, A=party[k], og_shape=og_shap[k],
                                         batch_size=batch_size, M=M[k], K=K, N=N[k])
    return result

###########################################onlyConvert############################################################
def convert_matric_batch(A, batch_size=4, M=8, K=6, N=30):
    A = A.cpu().numpy().astype(np.float64)
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)
    A = np.reshape(A, (1, -1)) 
    A = np.squeeze(A)
    A_len = len(A)
    # pad array at the end so tha the array is the size of
    A = A if (A_len % batch_size) == 0 \
        else np.pad(A, (0, batch_size - (A_len % batch_size)), 'constant', constant_values=(0, 0))
    idx_range = int(len(A) / batch_size) 
    batched_nums = []

    new_arr = np.array_split(A, idx_range)  
    for i in range(idx_range):
        batched_one = splicing(new_arr[i])
        batched_nums.append(batched_one)
    batched_nums = np.array(batched_nums)
    return batched_nums, og_shape


def batch_convert_per_layer(party: dict, batch_size: int, M: dict, K: int, N: dict):
    result = {}
    og_shapes = {}
    for k in party.keys():
        enc, shape_ = convert_matric_batch(party[k], batch_size=batch_size, M=M[k],
                                           K=K, N=N[k])
        result[k] = enc
        og_shapes[k] = shape_
    return result, og_shapes


def restore_shape_convert(A, shape, batch_size, M, K, N):
    batch_bits = (M + N) * batch_size
    for i in range(len(A)):
        A[i] = binary_float_decimal.dsb((str(A[i])))
    for i in range(len(A)):  
        if len(str(A[i])) < batch_bits:
            A[i] = '0' * (batch_bits - len(str(A[i]))) + str(A[i])
        elif len(str(A[i])) == batch_bits:
            A[i] = str(A[i])
        else:
            print("overflow:", type(A), A[i])
            warnings.warn('Overflow detected, consider using longer M,N')
    num_ele = np.prod(shape)
    num_ele_w_pad = batch_size * len(A)
    un_batched_nums = []
    for t in range(len(A)):
        move = 0
        for j in range(batch_size):
            tail = move + M + N
            un_batched_nums.append(A[t][move:tail])
            move = tail
    un_batched_nums = np.array(un_batched_nums)
    un_batched_nums_2_str = Parallel(n_jobs=N_JOBS)(
        delayed(binary_float_decimal.b2f)(i, M, K, N) for i in un_batched_nums)
    un_batched_nums_2_str = np.array(un_batched_nums_2_str).astype(np.float64)
    un_batched_nums_2_str = Parallel(n_jobs=N_JOBS)(
        delayed(float)(i) for i in un_batched_nums_2_str)
    res = np.reshape(un_batched_nums_2_str[0:num_ele], shape)
    return res


def de_convert_matrix_batch(A, og_shape, batch_size=4, M=8, K=6, N=30):
    A = np.array(A)
    result = restore_shape_convert(A, og_shape, batch_size, M, K, N)
    return result


def batch_de_convert_per_layer(party: dict, og_shape: dict, batch_size: int, M: dict, K: int, N: dict):
    result = {}
    for k in party.keys():
        result[k] = de_convert_matrix_batch(party[k], og_shape[k], batch_size=batch_size, M=M[k], K=K, N=N[k])
    return result


if __name__ == '__main__':
    array_A = {'weight': np.array([1.2, 2.5, 3.9]), 'bias': np.array([4.5, 5.6, 6.9])}
    B = {'weight': np.array([1.3, 2.1, 3.0]), 'bias': np.array([4.1, 5.1, 6.0])}
    encry_A, ogshape = batch_encrypt_per_layer(public_key, array_A, batch_size=2, M=3, K=1, N=5)
    print(encry_A)
    decry_A = batch_decrypt_per_layer(private_key, encry_A, ogshape, batch_size=2, M=3, K=1, N=5)
    print(decry_A)