#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class Solution {
public:
    int solve(int i,int j,int n,string & s,string & t ,vector<vector<int>> & dp){
        if(i==n||j==n){
            return 0;
        }
        if(s[i]==t[j]){
            return dp[i][j]= 1+ solve(i+1,j+1,n,s,t,dp);
        }
        if(dp[i][j]!=-1) return dp[i][j];
        return dp[i][j]= max(solve(i+1,j,n,s,t,dp),solve(i,j+1,n,s,t,dp));
    }
    int minInsertions(string s) {
        string t = s ;
        reverse(s.begin(),s.end());
        int n = s.size();
        vector<vector<int>> dp(n,vector<int>(n,-1));
        return n-solve(0,0,n,s,t,dp);
    }
};