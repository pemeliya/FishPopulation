
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include "omp.h"

struct Simulator {

struct PointLog2d {
    float x, y, logx;
};

std::vector< PointLog2d > readCsv(const std::string& path) {

    std::ifstream ifs(path);

    if(!ifs.is_open()) {
        throw std::runtime_error("Unable to open the file: " + path);
    }

    std::string line;
    std::getline(ifs, line); // skip caption

    std::vector< PointLog2d > vec;
    vec.reserve(100);
    PointLog2d pt;
    int num; char c;
    while(std::getline(ifs, line)) {

        std::istringstream is(line);
        is >> num >> c >> pt.x >> c >> pt.y;
        vec.push_back(pt);
    }
    return vec;
}

struct FishAgesNum {
    int64_t Nout0;
    int64_t Nout1;
    int64_t Nout2;
};

using Real = double;

void runLambdaInternal(std::ofstream& ofs, const FishAgesNum& opts, Real lambda0, 
            Real MwFactor, bool isGroupI) {

    struct Params{
        Real beta0 = 0, lambda0 = 0, Mw0 = 0;
        Real beta1 = 0, Mw1 = 0;
        Real sum = std::numeric_limits< Real >::max();
    };
    constexpr int nmins = 50;
    std::vector<Params> g_mins;

    omp_set_num_threads(8);

    constexpr Real step = 0.001;
    constexpr Real lambdaMin = 0.02, lambdaMax = 0.98;
    constexpr Real MpMin = 0.02, MpMax = 0.98;
    constexpr Real MwMin = 0.02, MwMax = 0.98;
    constexpr Real betaMinDiff = 0.02, lambdaMinDiff = 0.02;

    const char *group = isGroupI ? "I" : "II";
    fprintf(stderr, "\n--------- simulating group %s for lambda0 = %f ---------\n",
            group, lambda0);

    #pragma omp parallel shared(g_mins)
    {

    int num = (int)((1.0 - step*2) / step);
    Params cur, mins[nmins];
    cur.lambda0 = lambda0;

    #pragma omp for
    for(int i = 0; i < num; i++) {
    //for(cur.beta0 = step; cur.beta0 <= mod - step; cur.beta0 += step, num--) {
    cur.beta0 = step + step*i;

    if(i % 10 == 0)
        fprintf(stderr, "thid: %d; beta0: %.3f -- num: %d\n", omp_get_thread_num(), cur.beta0, i);

    //for(cur.lambda0 = 0.24; cur.lambda0 <= 0.24; cur.lambda0 += step)

    // original:
    // Mp + Mw0 + beta0 + lambda0 = 1
    // Mw1 + beta1 + lambda1 = 1
    // N(lambda2) = N(lambda1)*beta1/lambda1
    // N(lambda1)/lambda1 = N(lambda0)*beta0/lambda0
    // extra: lambda0 < lambda1
    // Mw0*5 = Mw1
    // N(lambda0)*beta0/lambda0 = N(Mw1) + N(lambda1)*beta1/lambda1 + N(lambda1)

    {
    for(cur.Mw0 = MwMin; cur.Mw0 <= MwMax; cur.Mw0 += step)
    {
        Real Mp = 1.0 - cur.beta0 - cur.lambda0 - cur.Mw0;
        if(!(Mp >= MpMin && Mp <= MpMax))
            continue;

        cur.Mw1 = cur.Mw0 / MwFactor;
        //for(cur.Mw1 = MwMin; cur.Mw1 <= cur.Mw0 / MwFactor; cur.Mw1 += step) 
        {
        
        for(cur.beta1 = step; cur.beta1 <= 1.0 - step; cur.beta1 += step) {

            Real lambda1 = 1.0 - cur.beta1 - cur.Mw1;
            if(!(cur.lambda0 + lambdaMinDiff < lambda1 &&
                 lambda1 >= lambdaMin && lambda1 <= lambdaMax))
                continue;

            if(!(cur.beta1 > cur.beta0 + betaMinDiff))
                continue;

            auto val1 = opts.Nout0*cur.beta0*lambda1 - opts.Nout1*cur.lambda0,
                  val2 = opts.Nout1*cur.beta1 - opts.Nout2*lambda1;
            cur.sum = std::abs(val1) + std::abs(val2);

            for(int i = 0; i < nmins; i++) {
                if(mins[i].sum > cur.sum) {
//                    memmove(mins + i + 1, mins + i, (nmins-1 - i)*sizeof(Params));
                    for(int j = nmins-1; j > i; j--) {
                        mins[j] = mins[j-1];
                    }
                    mins[i] = cur;
                    break;
                }
            } // for i
        }// for beta
        } // for Mw1
    } // for Mw0
    }
    } // for beta0

    #pragma omp critical
    {
        std::vector< Params > zmins(nmins + g_mins.size());
        std::merge(mins, mins + nmins, g_mins.begin(), g_mins.end(), zmins.begin(), 
                [](const Params& p1, const Params& p2)
        {
            return p1.sum < p2.sum;
        });
        g_mins.swap(zmins); // keep only the smallest ones..
    }

    } // parallel clause

    int i = 1; (void)i;
    std::sort(g_mins.begin(), g_mins.end(), [](const Params& p1, const Params& p2){
        return p1.beta0 < p2.beta0;
    });

    struct Range {
        Real rmin = std::numeric_limits< Real >::max();
        Real rmax = std::numeric_limits< Real >::min();
        Real ravg = 0;
        int total = 0;
    };

    auto setMM = [](Range& r, Real val){
        if(r.rmin > val)
            r.rmin = val;
        if(r.rmax < val)
            r.rmax = val;
        r.ravg += val;
        r.total++;
    };

    Range beta0R, beta1R, Mw0R, Mw1R, lambda0R, lambda1R, val1R, val2R;

    for(const Params& item: g_mins) {
        Real lambda1 = 1.0 - item.beta1 - item.Mw1;

        auto val1 = std::abs(opts.Nout0*item.beta0/item.lambda0 - opts.Nout1/lambda1),
             val2 = std::abs(opts.Nout1*item.beta1/lambda1 - opts.Nout2);

        auto Mp0 = 1.0 - item.beta0 - item.lambda0 - item.Mw0;
        if(item.beta0 == 0.0)
            continue;

        setMM(beta0R, item.beta0);
        setMM(beta1R, item.beta1);
        setMM(Mw0R, item.Mw0);
        setMM(Mw1R, item.Mw1);
        setMM(lambda0R, item.lambda0);
        setMM(lambda1R, lambda1);
        setMM(val1R, val1);
        setMM(val2R, val2);

        static char bbf[256];
        sprintf(bbf, "%d ; %.5f ; %.5f ; %.5f ; %.5f ; %.5f ; %.5f ; %.5f ; %.2f ; %.2f\n",
            i++, item.beta0, item.lambda0, Mp0, item.Mw0, item.beta1, lambda1, item.Mw1, val1, val2);
        ofs << bbf;
    }
}

void runLambda() 
{
    auto S = [](int64_t X) {
        Real mulC = 1.0 / 0.41;
        return (int64_t)(X * mulC);
    };
    FishAgesNum groupI = {S(18831), S(2634), S(1024)};
    FishAgesNum groupII = {S(3648), S(1107), S(431)};
    
    const std::array< int, 2 > numFish[] = {
     {198, 113},
     {336, 106},
     {198, 100},
     {296, 94},
     {198, 88},
     {257, 81},
     {198, 75},
     {217, 69},
     {198, 63},
     {178, 56},
     {158, 50},
     {138, 44},
     {119, 38},
     {99, 31},
     {79, 25},
     {59, 19},
     {40, 13},
    };

    int i = 0;
    for(auto[numFish0, numFish1] : numFish) 
    for(Real mwFactor = 4; mwFactor <= 7; mwFactor += 0.5, i++)
    {
        static char buf[256];

        {
        Real numFishEggs = 1000*0.95;   // the number of fish eggs survived from one female fish
        Real total = numFish0*numFishEggs;
        Real lambda0_I = groupI.Nout0 / total;

        sprintf(buf, "%d_FishDistr_%d_%.1f_I.csv", i, numFish0, mwFactor);

        std::ofstream ofs(buf);
        ofs << "sep=;\n";
        ofs << "GroupI; Nout0=" << groupI.Nout0 << "; Nout1=" << groupI.Nout1 << "; Nout2=" << groupI.Nout2 
                << "; mWFactor=" << mwFactor << "\n";
        ofs << "No; beta0 ; lambda0 ; Mp0; Mw0 ; beta1 ; lambda1 ; Mw1 ; |A1-A2| ; |B1-B2|\n";
        runLambdaInternal(ofs, groupI, lambda0_I, mwFactor, true);
        }

        {
        Real numFishEggs = 2000*0.95;     // the number of fish eggs survived from one female fish
        Real total = numFish1*numFishEggs;
        Real lambda0_II = groupII.Nout0 / total;

        sprintf(buf, "%d_FishDistr_%d_%.1f_II.csv", i, numFish1, mwFactor);
        std::ofstream ofs(buf);
        ofs << "sep=;\n";
        ofs << "GroupII; Nout0=" << groupII.Nout0 << "; Nout1=" << groupII.Nout1 << "; Nout2=" << groupII.Nout2 
                << "; mWFactor=" << mwFactor << "\n";
        ofs << "No; beta0 ; lambda0 ; Mp0; Mw0 ; beta1 ; lambda1 ; Mw1 ; |A1-A2| ; |B1-B2|\n";
        runLambdaInternal(ofs, groupII, lambda0_II, mwFactor, false);
        }
    } 

    exit(1);
}

}; // Simulator

int main(int argc, char *argv[])
{
    try {
        Simulator simi;
        simi.runLambda();
    }
    catch(std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
    }
    return 0;
}
