#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    void solve(int i ,int j, vector<string> & ans , string & temp ){
        if(i==0 && j==0){
            ans.push_back(temp);
            return;
    
        }
        //  here i is for '(' and j for ')'
        if(i >0){
            temp.push_back('(');
            solve(i-1,j,ans,temp);
            temp.pop_back();
            }
        // if we have more ')' than '(' then we can add ')'
        // this is the condition to check if we can add ')'
        if(j>i){
            temp.push_back(')');
            solve(i,j-1,ans,temp);
            temp.pop_back();
        }

    }
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        string temp = "";
        
        solve(n,n,ans ,temp );
        return ans;
    }
};