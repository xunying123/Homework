import hashlib
import random
from sympy import isprime
import base64


# 找到最大公约数
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# 扩展欧几里得算法
def extended_gcd(a, b):
    x, y, u, v = 0, 1, 1, 0
    while a != 0:
        q, r = b // a, b % a
        m, n = x - u * q, y - v * q
        b, a, x, y, u, v = a, r, u, v, m, n
    gcd = b
    return gcd, x, y
# 求模逆
def modinv(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m

# 生成大素数
def generate_large_prime(bits):
    while True:
        p = random.getrandbits(bits)
        if isprime(p):
            return p

# RSA签名类
class RSASignature:
    def __init__(self, bits=1024):
        self.p = generate_large_prime(bits)
        self.q = generate_large_prime(bits)
        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)
        self.e = 65537  # 公钥
        self.d = modinv(self.e, self.phi)  # 私钥

    # 对消息进行签名
    def sign(self, message):
        hashed_message = int.from_bytes(hashlib.sha256(message.encode()).digest(), byteorder='big')


        return signature

    # 验证签名
    def verify(self, message, signature):
        hashed_message = int.from_bytes(hashlib.sha256(message.encode()).digest(), byteorder='big')
        decrypted_signature = pow(signature, self.e, self.n)
        return hashed_message == decrypted_signature

# ElGamal加密类  
class ElGamalEncryption:
    # 生成ElGamal密钥对
    def __init__(self, bits = 1024):
        self.p =  generate_large_prime(bits) #大素数
        self.g = random.randint(2, self.p - 1) #生成元
        self.x = random.randint(2, self.p - 2) #私钥
        self.y = pow(self.g, self.x, self.p) #公钥     

    # 加密
    def elgamal_encrypt(self,plain_text):
        M = int.from_bytes(plain_text.encode(), byteorder='big')


        return C1, C2

    # 解密
    def elgamal_decrypt(self,C1, C2):
        s = pow(C1, self.x, self.p)
        M = (C2 * modinv(s, self.p)) %self.p
        return M.to_bytes((M.bit_length() + 7) // 8, byteorder='big')

# 请填充 RSASignature.sign 和ElGamalEncryption.elgamal_encrypt 中的代码，对"姓名+学号"组成的消息进行RSA签名和ElGamal加密，并能顺利通过签名验证、解密。
rsa = RSASignature()
message = "name(pinyin)+student_id"
signature = rsa.sign(message)
print("Signature:", signature)

# 验证签名
print("Verification:", rsa.verify(message, signature))

# 对消息进行ElGamal 加密
elgamal = ElGamalEncryption()
C1, C2 = elgamal.elgamal_encrypt(message)

print("Encrypted Message (C1, C2):", C1, C2)

# 解密
decrypted_message = elgamal.elgamal_decrypt(C1, C2)

# 验证消息
print("Decrypted Message:", decrypted_message.decode())
