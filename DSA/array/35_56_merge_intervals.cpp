#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ans;
        sort(intervals.begin(),intervals.end());
        vector<int> temp = intervals[0];
        int n = intervals.size();
        if(n==1) return intervals;
        for(int i=1;i<n;i++){
            if(intervals[i][0]>temp[1]){
                ans.push_back(temp);
                temp=intervals[i];
            }
            else{
                temp[1]= max(intervals[i][1],temp[1]);
            }
        }
        ans.push_back(temp);
        return ans;
    }
};