/*
	Authors: 
	
	ECE 462/562: Computer Architecture and Design
		Course Project
		Dr. Tosiron Adegbija
 */

/**
 * @file
 * Implements a VLDP prefetcher
   VLD prefetcher template instantiations
 */
 
 /* Mike's notes:
 So essentially how gem5 does prefetching is that the calculatePrefetch() function is called when a new memory access is recieved by the processor,
 whether that be a read or write request. Inside the calculatePrefetch() command, pkt->getAddr() is the actual address of the request.
 
 At this point in the function, you can do anything you want at your disposal to calculate the next address guess. If you have a guess, you put the integer address
 you're guessing into the prefetch queue by 
 
 Addr new_addr = address_guess;
 addresses.push_back(AddrPriority(new_addr, 0)); //this should happen within calculatePrefetch()
 
 So, a constructor for this class, the calculatePrefetch function, and the params constructor (the last function in this file) are the only things
 you actually need to create a prefetcher.
 
 This file is a copy of the stride prefetcher built into gem5, which is the closest prefetcher type to VLDP and I  would imagine would share a lot of the code,
 which is why I decided to base the vldp.cc file on stride.cc as a template. I already changed all instances of the stride prefetcher namesake to VLDP, so this
 file is ready to go to compile in gem5 as a vldp prefetcher
 
 So, basically this file needs to be modified to implement vldp. Any functions not listed eariler can be removed if you don't need them, or any new ones can be added
 
 If/when you change functions, you also need to update the vldp.hh file (header file, just put in the new function prototypes)

 
 VLDP is introduced and explained in this file: https://www.cs.utah.edu/~rajeev/pubs/micro15m.pdf
 
 From what I can tell, the biggest thing is that you need to do is use 3 prefetching tables that look further back, instead of just 1
 
 It's probably a good idea to get the logic of it working in your own Visual Studio project first by feeding it in an address list and generating your guess,
 and once that works bringing it inside here. Either way, Ill handle the compile and gem5 integration when you finish.
 
 
 */

#include "mem/cache/prefetch/vldp.hh"

#include "base/random.hh"
#include "base/trace.hh"
#include "debug/HWPrefetch.hh"

#include <iostream>
#include <string>
#include <fstream>

VLDPrefetcher::VLDPrefetcher(const VLDPrefetcherParams *p)
    : QueuedPrefetcher(p),
      maxConf(p->max_conf),
      threshConf(p->thresh_conf),
      minConf(p->min_conf),
      startConf(p->start_conf),
      pcTableAssoc(p->table_assoc),
      pcTableSets(p->table_sets),
      useMasterId(p->use_master_id),
      degree(p->degree),
      pcTable(pcTableAssoc, pcTableSets, name())
{
    // Don't consult stride prefetcher on instruction accesses
    onInst = false;

    assert(isPowerOf2(pcTableSets));
}

VLDPrefetcher::StrideEntry**
VLDPrefetcher::PCTable::allocateNewContext(int context)
{
    auto res = entries.insert(std::make_pair(context,
                              new StrideEntry*[pcTableSets]));
    auto it = res.first;
    chatty_assert(res.second, "Allocating an already created context\n");
    assert(it->first == context);

    DPRINTF(HWPrefetch, "Adding context %i with stride entries at %p\n",
            context, it->second);

    StrideEntry** entry = it->second;
    for (int s = 0; s < pcTableSets; s++) {
        entry[s] = new StrideEntry[pcTableAssoc];
    }
    return entry;
}

VLDPrefetcher::PCTable::~PCTable() {
    for (auto entry : entries) {
        for (int s = 0; s < pcTableSets; s++) {
            delete[] entry.second[s];
        }
        delete[] entry.second;
    }
}

void
VLDPrefetcher::calculatePrefetch(const PacketPtr &pkt,
                                    std::vector<AddrPriority> &addresses)
{
    if (!pkt->req->hasPC()) {
        DPRINTF(HWPrefetch, "Ignoring request with no PC.\n");
        return;
    }
	std::ofstream addressfile;
    // Get required packet info
    Addr pkt_addr = pkt->getAddr();
    Addr pc = pkt->req->getPC();
    bool is_secure = pkt->isSecure();
    MasterID master_id = useMasterId ? pkt->req->masterId() : 0;

    // Lookup pc-based information
    StrideEntry *entry;

    if (pcTableHit(pc, is_secure, master_id, entry)) {
        // Hit in table
        int new_stride = pkt_addr - entry->lastAddr;
        bool stride_match = (new_stride == entry->stride);

        // Adjust confidence for stride entry
        if (stride_match && new_stride != 0) {
            if (entry->confidence < maxConf)
                entry->confidence++;
        } else {
            if (entry->confidence > minConf)
                entry->confidence--;
            // If confidence has dropped below the threshold, train new stride
            if (entry->confidence < threshConf)
                entry->stride = new_stride;
        }

        DPRINTF(HWPrefetch, "Hit: PC %x pkt_addr %x (%s) stride %d (%s), "
                "conf %d\n", pc, pkt_addr, is_secure ? "s" : "ns", new_stride,
                stride_match ? "match" : "change",
                entry->confidence);

        entry->lastAddr = pkt_addr;

        // Abort prefetch generation if below confidence threshold
        if (entry->confidence < threshConf)
            return;

        // Generate up to degree prefetches
        for (int d = 1; d <= degree; d++) {
			
            // Round strides up to atleast 1 cacheline
            int prefetch_stride = new_stride;
            if (abs(new_stride) < blkSize) {
                prefetch_stride = (new_stride < 0) ? -blkSize : blkSize;
            }

            Addr new_addr = pkt_addr + d * prefetch_stride;
            if (samePage(pkt_addr, new_addr)) {
                DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
                addresses.push_back(AddrPriority(new_addr, 0));
				
				addressfile.open("address_list.txt", std::ios_base::app);
				addressfile << "Adding new address to table: ";
				addressfile << std::to_string(new_addr);
				addressfile << " The prefetch stride was: ";
				addressfile << std::to_string(prefetch_stride);
				addressfile << "\n";
				addressfile.close();
            } else {
                // Record the number of page crossing prefetches generated
                pfSpanPage += degree - d + 1;
                DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
                return;
            }
        }
    } else {
        // Miss in table
        DPRINTF(HWPrefetch, "Miss: PC %x pkt_addr %x (%s)\n", pc, pkt_addr,
                is_secure ? "s" : "ns");

        StrideEntry* entry = pcTableVictim(pc, master_id);
        entry->instAddr = pc;
        entry->lastAddr = pkt_addr;
        entry->isSecure= is_secure;
        entry->stride = 0;
        entry->confidence = startConf;
    }
}

inline Addr
VLDPrefetcher::pcHash(Addr pc) const
{
    Addr hash1 = pc >> 1;
    Addr hash2 = hash1 >> floorLog2(pcTableSets);
    return (hash1 ^ hash2) & (Addr)(pcTableSets - 1);
}

inline VLDPrefetcher::StrideEntry*
VLDPrefetcher::pcTableVictim(Addr pc, int master_id)
{
    // Rand replacement for now
    int set = pcHash(pc);
    int way = random_mt.random<int>(0, pcTableAssoc - 1);

    DPRINTF(HWPrefetch, "Victimizing lookup table[%d][%d].\n", set, way);
    return &pcTable[master_id][set][way];
}

inline bool
VLDPrefetcher::pcTableHit(Addr pc, bool is_secure, int master_id,
                             StrideEntry* &entry)
{
    int set = pcHash(pc);
    StrideEntry* set_entries = pcTable[master_id][set];
    for (int way = 0; way < pcTableAssoc; way++) {
        // Search ways for match
        if (set_entries[way].instAddr == pc &&
            set_entries[way].isSecure == is_secure) {
            DPRINTF(HWPrefetch, "Lookup hit table[%d][%d].\n", set, way);
            entry = &set_entries[way];
            return true;
        }
    }
    return false;
}

VLDPrefetcher*
VLDPrefetcherParams::create()
{
    return new VLDPrefetcher(this);
}
