#include <string>
#include <vector>   
using namespace std;
// LeetCode Problem: https://leetcode.com/problems/longest-common-prefix/
// Problem Statement: Write a function to find the longest common prefix string amongst an array of strings.
// If there is no common prefix, return an empty string "".
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        string ans = "";

        for(int i=0;i<strs[0].size();i++){
            char check=strs[0][i];
            for(int j=0;j<strs.size();j++){
                if(strs[j].size()<i){
                    return ans;
                }
                else if (strs[j][i] != check){
                    return ans;
                }
            }
            ans+=check;
        }
        return ans;
    }
};



