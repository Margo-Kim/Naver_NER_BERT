# from collectinos import defaultdict
# from pprint import pprint

# # 그 키값이 없더라도 추가해서 직전에 키에 대한 거를 추가해서 



# from collections import Counter

# from pathlib import Path

# # 쉘파이 / 쉘스크립트


# argparse.Arggu,ent Parser

# argument 를 셀프


class Solution:
    def romanToInt(self, s: str) -> int:
        value = 0
        i = 0 

        while i < len(s):
            if i+1 < len(s):
           
                if s[i] + s[i+1] =='IV':
                    value += 4
                    i += 2 
                elif s[i] + s[i+1] =='IX':
                    value += 9
                    i += 2 
                elif s[i] + s[i+1] =='XL':
                    value += 40
                    i += 2 
                elif s[i] + s[i+1] =='XC':
                    value += 90
                    i += 2 
                elif s[i] + s[i+1] =='CD':
                    value += 400
                    i += 2 
                elif s[i] + s[i+1] =='CM':
                    value += 900
                    i += 2 
                else: 
                    if s[i] == 'I':
                        value += 1
                    elif s[i] == 'V':
                        value += 5
                    elif s[i] == 'X':
                        value += 10
                    elif s[i] == 'L':
                        value += 50
                    elif s[i] == 'C':
                        value += 100
                    elif s[i] == 'D':
                        value += 500
                    elif s[i] == 'M':
                        value += 1000
                
                    i += 1

        return value
        
s = 'III'
a = Solution()
result = a.romanToInt(s)

print(result)