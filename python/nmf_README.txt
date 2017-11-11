This script describes conceptual difference between the two nmf scripts, and why there exists two.

	- nmf_threads.py:
		Utilizes threads, which shares memory, making it memory efficient across threads.
		However, threads are blocked by the GIL, such that only one thread runs at a time.
		Since, the pandas operations apparently realeases the GIL (probably uses numpy) threads
		will perform parallel operations.
		
		Note that even though writing to files is an I/O operation, the decompression of the data
		is CPU bound which does not release GIL.
		
	- nmf_embarrasingly_parallel:
		This script runs the nmf optimization and the exporting of data in true parallel. However, it is done using a
		pool of workers, such that we do not start and kill forks twice.
		
		Since, the entire function and arguments passed to these forks are done using subprocesses, each fork has
		it's own namespace. Thus, the memory taken by these are multiplied by number of jobs running.
		
		
	Consequently:
		The threading script is less memory dependent, though still slower than the embarassingly parallel script.
		For this exact situation the difference is not huge, and both can probably be utilized without considering
		the trade-off
		
