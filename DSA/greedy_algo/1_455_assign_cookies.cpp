#include <vector>
#include <algorithm>
using namespace std;
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(),g.end());
        sort(s.begin(),s.end());
        int gi =0;
        int ci=0;
        int m = g.size();
        int n = s.size();
        while(gi<m && ci < n){
            if(g[gi]<=s[ci]){
                gi++;
            }
            ci++;
        }
        return gi;
    }
};