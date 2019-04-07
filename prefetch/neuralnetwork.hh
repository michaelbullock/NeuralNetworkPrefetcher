/*
 * Authors:
 */

/**
 * @file
 * Implements a basic neural network prefetcher
 */

#ifndef __MEM_CACHE_PREFETCH_NEURALNETWORK_HH__
#define __MEM_CACHE_PREFETCH_NEURALNETWORK_HH__

// Extends the queued prefetching scheme

#include <iostream>
#include <string>
#include <fstream>


#include "mem/cache/prefetch/queued.hh"
#include "params/NeuralNetworkPrefetcher.hh"

class NeuralNetworkPrefetcher : public QueuedPrefetcher
{
  //define neural network specific functions and variables here. These are for internal prefetcher use, not for GEM5 visibility
  protected:
	std::ofstream addressfile;
	int totalCalcs;

  //inherited from queued prefetcher
  public:

    NeuralNetworkPrefetcher(const NeuralNetworkPrefetcherParams *p);

    void calculatePrefetch(const PacketPtr &pkt,
                           std::vector<AddrPriority> &addresses);
};

#endif // __MEM_CACHE_PREFETCH_NEURALNETWORK_HH__
