import pyotp
code = "YEL7FTSOYPYBCVIN"

totp = pyotp.TOTP(code)
print totp.now()

# def googleauthcode(secret):
# 	key = bytearray(base64.b32decode(secret))
# 	curtime = bytearray(int(time.time()) / 30)
# 	concat = key + curtime
# 	print len(concat)

# googleauthcode(code)
