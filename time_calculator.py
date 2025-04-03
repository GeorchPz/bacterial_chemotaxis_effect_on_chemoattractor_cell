def mins2time(mins):
    d, rem = divmod(mins, 24*60)
    h, min = divmod(rem, 60)
    return "{}d {:0>2}h{:0>2}min".format(int(d), int(h), int(min))

def secs2time(secs):
    m, s = divmod(secs, 60)
    return "{}{:0>2}sec".format(mins2time(m), int(s))

def main():
    print(
        f'from {mins2time(250**2*0.33)} to {mins2time(250**2/2*0.01)}',
        f'from {mins2time(2997)} to {secs2time(7)}',
        sep='\n'
    )

if __name__ == '__main__':
    main()