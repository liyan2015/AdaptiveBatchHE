from functools import lru_cache
index_acc = 0
a = 0
index_i = 0
@lru_cache
def f2b(num, M=8, K=6, N=30):  
    '''
    floating number to binary
    '''
    global index_acc
    global a
    str_num = str(float(num))
    accuracy = K
    if '.' in str_num:  
        if num == 0:
            return '0' * (M + N)
        else:
            string_integer, string_decimal = str_num.split('.')
            integer = int(string_integer)
            integer2b = '{:b}'.format(integer).zfill(M)  
            lst_accuracy = []
            if len(string_decimal) >= accuracy:
                for i in range(accuracy):  
                    lst_accuracy.append(string_decimal[i])
            else:
                for i in string_decimal:
                    lst_accuracy.append(i)
                a = accuracy - len(string_decimal)
                for i in range(a):
                    lst_accuracy.append('0')  
            for i in lst_accuracy:  
                if i != '0':
                    index_acc = lst_accuracy.index(i)
                    break
            str1 = ''.join(lst_accuracy)
            str_float = str1[index_acc::]
            num_float = int(str_float)
            num_float2b = '{:b}'.format(num_float)  
            if len(str(num_float2b)) <= N:
                a = N - len(str(num_float2b))
            else:
                print('N error!!!')
            str_all = integer2b + '0' * a + str(num_float2b)
            return str_all  
    else:
        integer2b = '{:b}'.format(num).zfill(M) 
        str_all = integer2b + '0' * N
        return str_all  


@lru_cache
def b2f(num, M=8, K=6, N=30):
    '''
    binary to floating number
    '''
    accuracy = K
    bit = M + N
    if len(str(num)) == bit:
        str_front = str(num)[0:M]  
        str_back = str(num)[M:]  
        bin2int = int(str_front, 2)  
        for i in str_back: 
            if i == '1':
                index_float = str_back.index(i)
                break
            else:
                index_float = 0
        str_back_ex = str_back[index_float::]
        a = int(str_back_ex, 2)
        if accuracy >= len(str(a)):
            b = accuracy - len(str(a))
            str_all = str(bin2int) + '.' + '0' * b + str(a)
            return str_all
        else:
            if len(str(a)) == K + 1:
                a_real = str(a)[1:]
                str_all = str(bin2int + int(str(a)[0])) + '.' + str(a_real)
                return str_all
            else:
                print('b2f error!')
    else:
        print('num input error')

def dsb(c: str):
    '''
    Convert decrypted plaintext into binary
    '''
    string = list(c[::-1])
    for i in range(len(string) - 1):
        if int(string[i]) % 2 == 0 and int(string[i]) != 0:
            carry = int(string[i]) // 2
            string[i] = str(0)
            string[i + 1] = str(carry + int(string[i + 1]))
        elif int(string[i]) % 2 == 1 and int(string[i]) != 1:
            carry = int(string[i]) // 2
            string[i] = str(1)
            string[i + 1] = str(carry + int(string[i + 1]))
        else:
            continue
    last = int(string[-1])
    last_b = '{:b}'.format(last)[::-1]
    string[-1] = last_b[0]
    for i in range(1, len(last_b)):
        string.append(last_b[i])
    res = ''.join(string[::-1])
    return res

def integer_floating_dsb(string: str, M, N):  
    M_string = string[0:M]
    N_string = string[M:]
    M_dsb = dsb(M_string)
    N_dsb = dsb(N_string)
    return M_dsb + N_dsb


def total_bits(num: int) -> int:
    string = '{:b}'.format(num)
    return len(string)


def decode_fillzero(num:int, batch_length) -> str:
    decoded_num = '{:b}'.format(num).zfill(batch_length)
    return decoded_num

if __name__ == '__main__':
    a = f2b(0.)
    print(a)
    print(type(a))
    a_back = b2f(a)
    # b = f2b(2.49)
    # b_back = b2f(b)
    print(a_back)
    # print(type(a_back))
    # print(b_back)
    # # print('-----------------------------')
    # print(bin(15))
    # print(total_bits(15))