/*
	Authors: 

	
	ECE 462/562: Computer Architecture and Design
		Course Project
		Dr. Tosiron Adegbija
 */

/**
 * @file
 * Implements a VLD prefetcher
 */

#ifndef __MEM_CACHE_PREFETCH_VLDP_HH__
#define __MEM_CACHE_PREFETCH_VLDP_HH__

#include <unordered_map>

#include "mem/cache/prefetch/queued.hh"
#include "params/VLDPrefetcher.hh"

class VLDPrefetcher : public QueuedPrefetcher
{
  protected:
    const int maxConf;
    const int threshConf;
    const int minConf;
    const int startConf;

    const int pcTableAssoc;
    const int pcTableSets;

    const bool useMasterId;

    const int degree;

    struct StrideEntry
    {
        StrideEntry() : instAddr(0), lastAddr(0), isSecure(false), stride(0),
                        confidence(0)
        { }

        Addr instAddr;
        Addr lastAddr;
        bool isSecure;
        int stride;
        int confidence;
    };

    class PCTable
    {
      public:
        PCTable(int assoc, int sets, const std::string name) :
            pcTableAssoc(assoc), pcTableSets(sets), _name(name) {}
        StrideEntry** operator[] (int context) {
            auto it = entries.find(context);
            if (it != entries.end())
                return it->second;

            return allocateNewContext(context);
        }

        ~PCTable();
      private:
        const std::string name() {return _name; }
        const int pcTableAssoc;
        const int pcTableSets;
        const std::string _name;
        std::unordered_map<int, StrideEntry**> entries;

        StrideEntry** allocateNewContext(int context);
    };
    PCTable pcTable;

    bool pcTableHit(Addr pc, bool is_secure, int master_id, StrideEntry* &entry);
    StrideEntry* pcTableVictim(Addr pc, int master_id);

    Addr pcHash(Addr pc) const;
  public:

    VLDPrefetcher(const VLDPrefetcherParams *p);

    void calculatePrefetch(const PacketPtr &pkt,
                           std::vector<AddrPriority> &addresses);
};

#endif // __MEM_CACHE_PREFETCH_VLDP_HH__
