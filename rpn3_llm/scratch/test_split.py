

def split_rpn(s):
    parts = s.split("[ANS]")
    if len(parts) >= 2:
        ans = parts[-1].split("[EOS]")[0].strip()
        content = parts[0]
    else:
        ans = ""
        content = s
    
    rev_math = content.split("[REV]")
    if len(rev_math) < 2:
        return "", "", ans
    
    rev_math_content = rev_math[1]
    math_parts = rev_math_content.split("[MATH]", 1) # Split only at first [MATH]
    
    rev = math_parts[0] if len(math_parts) > 0 else ""
    math = math_parts[1] if len(math_parts) > 1 else ""
    
    return rev, math, ans

s = "[BOS]6672948955532 22+6795422462-?[REV]2355598492766 22+2642245976-=[MATH]2+2+0=4:3+2+0=5:5+0+0=5:5+0+0=5:5+0+0=5:9+0+0=9:8+0+0=8:4+0+0=4:9+0+0=9:2+0+0=2:7+0+0=7:6+0+0=6:6+0+0=6=4555598492766:4555598492766 2642245976-=[MATH]4-2-0=2:5-6-0=9:5-4-1=0:5-2-0=3:5-2-0=3:9-4-0=5:8-5-0=3:4-9-0=5:9-7-1=1:2-6-0=6:7-0-1=6:6-0-0=6:6-0-0=6:[BORROW]0|+:2903353516666=2903353516666[ANS]6666153533092[EOS]"
#s = "[BOS]8956699541284 2741723511659269+?[REV]4821459966598 9629561153271472+=[MATH]4+9+0=3:8+6+1=5:2+2+1=5:1+9+0=0:4+5+1=0:5+6+1=2:9+1+1=1:9+1+1=1:6+5+1=2:6+3+1=0:5+2+1=8:9+7+0=6:8+1+1=0:0+4+1=5:0+7+0=7:0+2+0=2=3550021120860572[ANS]2750680211200553[EOS]"
rev, math, ans = split_rpn(s)
math_parts = math.split("=")
print(f"REV: {rev}")
print(f"Math Parts: {math_parts}")
print(f"ANS: {ans}")