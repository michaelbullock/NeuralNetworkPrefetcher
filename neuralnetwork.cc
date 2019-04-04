/*
 * Authors: 
		Michael Bullock - bullockm@email.arizona.edu
		Michael Inouye - mikesinouye@email.arizona.edu
		Curt Bansil - curtbansil@email.arizona.edu
		Alli Gilbreath - alligilbreath@email.arizona.edu
	
	ECE 462/562: Computer Architecture and Design
		Course Project
		Dr. Tosiron Adegbija
 */

/**
 * @file
 * Implements a basic neural network prefetcher
   Neural network prefetcher template instantiations
 */

#include "mem/cache/prefetch/queued.hh"

#include "base/random.hh"
#include "base/trace.hh"
#include "debug/HWPrefetch.hh"

NeuralNetworkPrefetcher::NeuralNetworkPrefetcher(const NeuralNetworkPrefetcher *p) : QueuedPrefetcher(p), {
	
    // Don't consult the prefetcher on instruction accesses
    onInst = false;

}

void NeuralNetworkPrefetcher::calculatePrefetch(const PacketPtr &pkt, std::vector<AddrPriority> &addresses)
{
    if (!pkt->req->hasPC()) {
        DPRINTF(HWPrefetch, "Ignoring request with no PC.\n");
        return;
    }

    // Get required packet info
    Addr pkt_addr = pkt->getAddr();
    Addr pc = pkt->req->getPC();
    bool is_secure = pkt->isSecure();
    MasterID master_id = useMasterId ? pkt->req->masterId() : 0;

	
	
	
	/* Insert nn code here */
	
	
	
	
	
	Addr new_addr = 0; // This is where the magic happens
	
	// Check to see if the new address to prefetch is within the same page as the current packet address
	if (samePage(pkt_addr, new_addr)) {
		DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
		addresses.push_back(AddrPriority(new_addr, 0));	
	}
	
	else {
		DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
		return;
	}
	
	return;

}

NeuralNetworkPrefetcher*
NeuralNetworkPrefetcherParams::create()
{
    return new NeuralNetworkPrefetcher(this);
}
