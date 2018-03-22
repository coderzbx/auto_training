#!/usr/bin/env python

"""
PrefetchIter transforms a common iter to a multiprocessing version.
"""

from __future__ import absolute_import
import json
import logging
import argparse
import time
import numpy as np
import mxnet as mx
import multiprocessing as mp

class PrefetchIter(mx.io.DataIter):
    """Performs prefetch for other data iterator.

    This iterator will create multiple processes to perform data reading and
    store the data in buffer. It potentially accelerates data reading, at the
    cost of more memory usage.
    """

    def __init__(self, iter, threads=4, buffer_size=3):
        """Initializer.

        Args:
            iter : mx.io.DataIter, the data iterator to be prefetched.
            threads: int, number of processes.
            buffer_size: int, size of buffer for each process.
        """
        super(PrefetchIter, self).__init__(iter.batch_size)
        self._iter = iter
        self._nsamples = self._iter.size
        self._nprocs = threads
        self._buffer_size = buffer_size
        self._procs = []
        self._cursor = 0
        assert self._iter.size == len(self._iter.index), \
          "{}<=>{}".format(self._iter.size, len(self._iter.index))

    def __del__(self):
        """Destructor."""
        for proc in self._procs:
            proc.join()

    @property
    def batches_per_epoch(self):
        """Number of batches per epoch."""
        return self._iter.batches_per_epoch

    @property
    def provide_data(self):
        """The name and shape of data."""
        return self._iter.provide_data

    @property
    def provide_label(self):
        """The name and shape of label."""
        return self._iter.provide_label

    def reset(self):
        """Restart from the beginning."""
        # cleanup
        for proc in self._procs:
            proc.join()
        # restart
        self._iter.reset()
        m = mp.Manager()
        self._shared_queue = m.Queue()
        self._procs = []
        for i in xrange(self._nprocs):
            fetcher = Prefetcher(self._iter, self._shared_queue, self._nprocs, i)
            proc = mp.Process(target=fetcher, name='prefetcher-{}'.format(i))
            proc.daemon = True
            proc.start()
            self._procs.append(proc)
        self._cursor = 0

    def next(self):
        """Read a batch."""
        # check stop
        if self._cursor >= self._nsamples:
            raise StopIteration
        id = (self._cursor / self._iter.batch_size) % self._nprocs
        tic = time.time()
        all_data, all_label = self._shared_queue.get()
        batch_data = [mx.nd.array(all_data[key]) for key in self._iter.data_name]
        batch_label = [mx.nd.array(all_label[key]) for key in self._iter.label_name]
        provide_data = [(k, v.shape) for k, v in zip(self._iter.data_name, batch_data)]
        provide_label = [(k, v.shape) for k, v in zip(self._iter.label_name, batch_label)]
        batch = mx.io.DataBatch(data=batch_data,
                                label=batch_label,
                                pad=self.getpad(),
                                index=self.getindex(),
                                provide_data=provide_data,
                                provide_label=provide_label)
        logging.info('waited for {} seconds'.format(time.time() - tic))
        self._cursor += self._iter.batch_size
        return batch

    def getindex(self):
        return self._cursor / self._iter.batch_size

    def getpad(self):
        if self._cursor + self._iter.batch_size > self._nsamples:
            return self._cursor + self._iter.batch_size - self._nsamples
        else:
            return 0


class Prefetcher(object):
    """Process functor for PrefetchIter."""

    def __init__(self, iter, queue, threads, id):
        """Initializer.

        Args:
            iter : mx.io.DataIter, the data iterator to be prefetched.
            queue: multiprocessing.Queue, where the prefetched data stored.
            threads: int, number of processes.
            id: int, id of current process.
        """
        super(Prefetcher, self).__init__()
        self._iter = iter
        self._queue = queue
        self._nprocs = threads
        self._id = id

    def __call__(self):
        """Functor."""
        logger = mp.get_logger()
        cursor = self._iter.batch_size * self._id
        nsamples = len(self._iter.index)
        while cursor < nsamples:
            # Check ReadData Range
            cursor_to = min(cursor + self._iter.batch_size, nsamples)
            tic = time.time()
            # Read data
            all_data, all_label = self._iter.get_given_index_batch(cursor, cursor_to)
            indexes = [self._iter.index[i] for i in range(cursor, cursor_to)]
            self._queue.put((all_data, all_label))
            logger.info('{}, {}, {} seconds'.format(cursor, indexes, time.time() - tic))
            cursor += self._iter.batch_size * self._nprocs

