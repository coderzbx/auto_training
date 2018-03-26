import multiprocessing

manager = multiprocessing.Manager()
task_queue = manager.Queue()
cut_queue = manager.Queue()

divide_queue = manager.Queue()
extend_queue = manager.Queue()
