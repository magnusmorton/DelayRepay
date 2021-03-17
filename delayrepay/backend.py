import os

if 'DELAY_CPU' in os.environ:
    print('fooo')
    import delayrepay.cpu as be
elif 'DELAY_LIFT' in os.environ:
    import delayrepay.rise as be
else:
    import delayrepay.cuda as be

backend = be
