'''
Name: Beier (Benjamin) Liu
Date: 6/30/2018

Remark:
Python 3.6 is recommended
Before running please install packages *numpy, scipy, matplotlib
Using cmd line py -3.6 -m pip install [package_name]
'''
import functools
import time
import logging
import multiprocessing

'''===================================================================================================
File content:

==================================================================================================='''
def Timer(func):
	@functools.wraps(func)
	def wrapped(*args, **kwargs):
		s=time.time()
		res=func(*args, **kwargs);
		e=time.time()
		logging.info('{}: {} seconds.'.format(func, round(e-s, 4)))
		return res
	return wrapped

def memoize(func):
    memo = {}
    @functools.wraps(func)
    def wrapped(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapped

@Timer
def multiProcess(process_num, func, *args):
	input_queue=multiprocessing.Queue()
	output_queue=multiprocessing.Queue()

	for i in range(process_num):
		input_queue.put((func, tuple([a[i] for a in args]))) # ith elements from all params of args

	for i in range(process_num):
		multiprocessing.Process(target=doWork, args=(input_queue, output_queue)).start()

	res=[];

	while 1:
		r=output_queue.get()
		logging.debug('The r is {}'.format(r))
		if r!='Done':
			res.append(r)
		else :
			break
	input_queue.close(); output_queue.close()
	return res

@Timer
def doWork(input, output):
	while 1:
		try :
			func, args=input.get(timeout=3);
			res=func(*args)
			output.put(res)
		except Exception as e:
			logging.debug('{}'.format(e));
			output.put('Done')
			break
