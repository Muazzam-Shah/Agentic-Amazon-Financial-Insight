import os
import json
import pandas as pd
from typing import Dict, Any, List
# from openai import OpenAI  # Commented out OpenAI import

# New import for Google Generative AI (Gemini)
import google.generativeai as genai

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback

class AmazonTradingAnalyzer:
    """Basic data analysis engine for Amazon trading data"""
    
    def __init__(self, data_path: str):
        """Initialize the analyzer with data from file"""
        self.df = self._load_data(data_path)
        
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from Excel file"""
        if file_path.endswith('.xlsx'):
            return pd.read_excel(file_path,nrows=150)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path,nrows=150)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")
    
    def get_data_summary(self) -> str:
        """Get a basic summary of the dataset"""
        if self.df.empty:
            return "No data available."
        
        total_sales = self.df['SalesOrganic'].sum() if 'SalesOrganic' in self.df.columns else 0
        total_units = self.df['UnitsOrganic'].sum() if 'UnitsOrganic' in self.df.columns else 0
        total_refunds = self.df['Refunds'].sum() if 'Refunds' in self.df.columns else 0
        total_net_profit = self.df['NetProfit'].sum() if 'NetProfit' in self.df.columns else 0
        
        summary = f"""
Dataset Summary:
- Total Records: {len(self.df)}
- Total Organic Sales: ${total_sales:,.2f}
- Total Organic Units: {total_units:,}
- Total Refunds: ${total_refunds:,.2f}
- Total Net Profit: ${total_net_profit:,.2f}
        """
        return summary.strip()
    
    def analyze_sales_trends(self, period: str = "monthly") -> str:
        """Analyze sales trends over time"""
        if self.df.empty:
            return "No data available for trend analysis."
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        if period.lower() == "monthly":
            grouped = self.df.groupby(pd.Grouper(key='Date', freq='ME'))
        elif period.lower() == "weekly":
            grouped = self.df.groupby(pd.Grouper(key='Date', freq='W'))
        else:
            return f"Invalid period: {period}. Please use monthly or weekly."
        
        trends = grouped.agg({
            'SalesOrganic': 'sum',
            'UnitsOrganic': 'sum',
            'NetProfit': 'sum'
        }).reset_index()
        
        result = f"{period.capitalize()} Sales Trends:\n"
        for _, row in trends.tail(5).iterrows():
            result += f"Date: {row['Date'].strftime('%Y-%m-%d')}, Sales: ${row['SalesOrganic']:,.2f}, Units: {row['UnitsOrganic']:,}, Profit: ${row['NetProfit']:,.2f}\n"
        
        return result
    
    def top_performing_products(self, metric: str = "sales", limit: int = 5) -> str:
        """Find top performing products"""
        if self.df.empty:
            return "No data available for product analysis."
        
        if metric.lower() == "sales":
            metric_col = "SalesOrganic"
        elif metric.lower() == "profit":
            metric_col = "NetProfit"
        elif metric.lower() == "units":
            metric_col = "UnitsOrganic"
        else:
            return f"Invalid metric: {metric}. Please use sales, profit, or units."
        
        top_products = self.df.groupby(['ASIN', 'Name']).agg({
            metric_col: 'sum'
        }).reset_index().sort_values(by=metric_col, ascending=False).head(limit)
        
        result = f"Top {limit} Products by {metric.capitalize()}:\n"
        for _, row in top_products.iterrows():
            result += f"ASIN: {row['ASIN']}, Product: {row['Name'][:50]}..., {metric.capitalize()}: {row[metric_col]:,.2f}\n"
        
        return result
    
    def marketplace_comparison(self) -> str:
        """Compare performance across marketplaces"""
        if self.df.empty:
            return "No data available for marketplace comparison."
        
        marketplace_data = self.df.groupby('Marketplace').agg({
            'SalesOrganic': 'sum',
            'UnitsOrganic': 'sum',
            'NetProfit': 'sum'
        }).reset_index()
        
        result = "Marketplace Comparison:\n"
        for _, row in marketplace_data.iterrows():
            result += f"Marketplace: {row['Marketplace']}, Sales: ${row['SalesOrganic']:,.2f}, Units: {row['UnitsOrganic']:,}, Profit: ${row['NetProfit']:,.2f}\n"
        
        return result


class BasicAIAgent:

    def __init__(self, data_path: str, gemini_api_key: str = None, openai_api_key: str = None):
        """Initialize the basic AI agent with LangChain"""
        self.analyzer = AmazonTradingAnalyzer(data_path)
        
        # Set up Gemini
        gemini_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it directly.")
        
        genai.configure(api_key=gemini_key)
        
        # Initialize LangChain ChatGoogleGenerativeAI
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_key,
                temperature=0.1
            )
        except Exception as e:
            print(f"Error initializing model 'gemini-2.0-flash': {str(e)}")
            print("Trying fallback model...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.0-pro",
                google_api_key=gemini_key,
                temperature=0.1
            )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Create LangChain tools
        self.tools = self._create_langchain_tools()
        
        # Initialize the agent with better configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
            # max_iterations=5,  # Increased iterations
            # max_execution_time=60,  # Add time limit
            # early_stopping_method="generate"  # Better stopping
        )
        
        # output dir for Markdown files
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Analysis results will be saved to: {self.output_dir}")
        
        print("Basic AI Agent initialized successfully with LangChain and Gemini")
    
    def _create_langchain_tools(self) -> List[Tool]:
        """Create LangChain tools for data analysis"""
        tools = [
            Tool(
                name="get_data_summary",
                description="Get a comprehensive summary of the Amazon trading dataset including total sales, units, refunds, and net profit. Use this when user asks for overview, summary, or general data insights.",
                func=self._tool_get_summary
            ),
            Tool(
                name="analyze_sales_trends",
                description="Analyze sales trends over time. Input should be 'monthly' or 'weekly' for the time period. Use this when user asks about trends, patterns, or time-based analysis.",
                func=self._tool_analyze_trends
            ),
            Tool(
                name="get_top_products",
                description="Find top performing products. Input should be in format 'metric:limit' where metric is 'sales', 'profit', or 'units' and limit is a number (default 5). Use this when user asks about best products, top performers, or product rankings.",
                func=self._tool_top_products
            ),
            Tool(
                name="compare_marketplaces",
                description="Compare performance across different Amazon marketplaces. No input required. Use this when user asks about marketplace comparison or regional performance.",
                func=self._tool_marketplace_comparison
            ),
            Tool(
                name="save_analysis",
                description="Save the current analysis to a markdown file. Input should be the content to save. Use this after providing analysis results.",
                func=self._tool_save_analysis
            )
        ]
        return tools
    
    def _tool_get_summary(self, input_str: str = "") -> str:
        """LangChain tool wrapper for data summary"""
        try:
            return self.analyzer.get_data_summary()
        except Exception as e:
            return f"Error getting data summary: {str(e)}"
    
    def _tool_analyze_trends(self, period: str = "monthly") -> str:
        """LangChain tool wrapper for trend analysis"""
        try:
            period = period.strip().lower()
            if period not in ["monthly", "weekly"]:
                period = "monthly"
            return self.analyzer.analyze_sales_trends(period)
        except Exception as e:
            return f"Error analyzing trends: {str(e)}"
    
    def _tool_top_products(self, input_str: str = "sales:5") -> str:
        """LangChain tool wrapper for top products analysis"""
        try:
            parts = input_str.split(":")
            metric = parts[0].strip().lower() if len(parts) > 0 else "sales"
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            
            if metric not in ["sales", "profit", "units"]:
                metric = "sales"
            
            return self.analyzer.top_performing_products(metric, limit)
        except Exception as e:
            return f"Error getting top products: {str(e)}"
    
    def _tool_marketplace_comparison(self, input_str: str = "") -> str:
        """LangChain tool wrapper for marketplace comparison"""
        try:
            return self.analyzer.marketplace_comparison()
        except Exception as e:
            return f"Error comparing marketplaces: {str(e)}"
    
    def _tool_save_analysis(self, content: str) -> str:
        """LangChain tool wrapper for saving analysis"""
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"langchain_analysis_{timestamp}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Analysis saved to: {filename}"
        except Exception as e:
            return f"Error saving analysis: {str(e)}"
    
    def _save_to_markdown(self, content: str, query: str, tool_used: str = None) -> str:
        """Save analysis results to a Markdown file with proper formatting"""
        try:
            # Create timestamp for filename
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a sanitized filename from the query
            query_part = "".join(c for c in query[:30] if c.isalnum() or c.isspace()).strip().replace(" ", "_")
            filename = f"{timestamp}_{query_part}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            # Format the content as markdown
            timestamp_readable = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            title = f"# Amazon FBA Analysis: {query}\n\n"
            metadata = f"*Generated on: {timestamp_readable}*\n\n"
            
            if tool_used:
                metadata += f"*Analysis type: {tool_used}*\n\n"
            
            separator = "---\n\n"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # f.write(title)
                f.write(metadata)
                f.write(separator)
                f.write(content)
            
            return filepath
        except Exception as e:
            print(f"Warning: Could not save markdown file: {str(e)}")
            return None

    def query(self, user_query: str) -> str:
        """Process user query using LangChain agent with fallback"""
        try:
            # Enhanced system prompt for the agent
            system_prompt = """You are an expert Amazon FBA business analyst. Use the available tools to analyze data and provide clear, actionable insights.

AVAILABLE TOOLS:
- get_data_summary: Overview of dataset metrics
- analyze_sales_trends: Time-based sales analysis  
- get_top_products: Best performing products
- compare_marketplaces: Regional performance comparison

INSTRUCTIONS:
1. Use 1-2 relevant tools to gather data
2. Provide clear analysis with specific numbers
3. Give actionable business recommendations
4. Keep responses concise and focused

Format your final response in markdown with key findings and recommendations."""

            # Create a comprehensive prompt
            full_prompt = f"{system_prompt}\n\nUser Question: {user_query}"
            
            try:
                # Use invoke instead of deprecated run method
                response = self.agent.invoke({"input": full_prompt})
                
                # Extract the output from the response
                if isinstance(response, dict):
                    agent_output = response.get("output", str(response))
                else:
                    agent_output = str(response)
                
            except Exception as agent_error:
                print(f"Agent execution failed: {str(agent_error)}")
                # Fallback to direct tool usage
                agent_output = self._fallback_analysis(user_query)
            
            # Format the response and save to markdown
            formatted_response = self._format_agent_response(agent_output, user_query)
            
            # Save the analysis
            saved_path = self._save_to_markdown(formatted_response, user_query, "langchain_agent")
            if saved_path:
                formatted_response += f"\n\n---\n\n*Analysis saved to: {os.path.basename(saved_path)}*"
            
            return formatted_response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def _fallback_analysis(self, user_query: str) -> str:
        """Fallback analysis when agent fails"""
        try:
            query_lower = user_query.lower()
            results = []
            
            # Determine which tools to use based on keywords
            if any(word in query_lower for word in ['summary', 'overview', 'total']):
                results.append("## Data Summary\n" + self._tool_get_summary())
            
            if any(word in query_lower for word in ['trend', 'seasonal', 'time', 'month']):
                results.append("## Sales Trends\n" + self._tool_analyze_trends("monthly"))
            
            if any(word in query_lower for word in ['top', 'best', 'product', 'asin']):
                results.append("## Top Products\n" + self._tool_top_products("sales:5"))
            
            if any(word in query_lower for word in ['marketplace', 'region', 'compare']):
                results.append("## Marketplace Comparison\n" + self._tool_marketplace_comparison())
            
            # If no keywords matched, provide summary
            if not results:
                results.append("## Data Summary\n" + self._tool_get_summary())
            
            fallback_response = "\n\n".join(results)
            fallback_response += "\n\n## Analysis Notes\n*This analysis was generated using fallback mode due to agent processing limitations.*"
            
            return fallback_response
            
        except Exception as e:
            return f"Fallback analysis failed: {str(e)}"
    
    def _format_agent_response(self, response: str, query: str) -> str:
        """Format the agent response with proper markdown structure"""
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        formatted = f"""# Amazon FBA Analysis: {query}

*Generated on: {timestamp} using LangChain Agent*

---

{response}

---

## 📊 Analysis Summary
This analysis was performed using LangChain tools with Gemini AI to provide comprehensive business insights for your Amazon FBA operations.
"""
        return formatted
    
    def get_conversation_history(self) -> str:
        """Get the conversation history from memory"""
        try:
            messages = self.memory.chat_memory.messages
            history = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                    history.append(f"{role}: {msg.content}")
            return "\n".join(history)
        except Exception as e:
            return f"Error retrieving conversation history: {str(e)}"
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        print("Conversation memory cleared.")

def main():
    """Main function to run the Basic AI Agent with LangChain"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python basic_ai_agent.py <data_file.xlsx>")
        return
    
    data_file = sys.argv[1]
    
    gemini_api_key = None
    try:
        api_key_file = os.path.join(os.path.dirname(__file__), "Gemeni_API_KEY")
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r') as f:
                gemini_api_key = f.read().strip()
                print("Gemini API key loaded from file.")
    except Exception as e:
        print(f"Error reading API key file: {str(e)}")
    
    if not gemini_api_key:
        gemini_api_key = input("Enter your Gemini API key (or press Enter if set in environment): ").strip()
        if not gemini_api_key:
            gemini_api_key = None
    
    try:
        agent = BasicAIAgent(data_path=data_file, gemini_api_key=gemini_api_key)
        
        print("\n🤖 LangChain Amazon Trading Analysis AI Agent (Powered by Gemini)")
        print("Available commands: summary, trends, top products, marketplace comparison")
        print("Special commands: 'history' (show conversation), 'clear' (clear memory)")
        print(f"Results will be saved to: {agent.output_dir}")
        print("Type 'exit' to quit\n")
        
        while True:
            user_input = input("Query: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            elif user_input.lower() == 'history':
                print("\nConversation History:")
                print(agent.get_conversation_history())
                continue
            elif user_input.lower() == 'clear':
                agent.clear_memory()
                continue
                
            print("\nProcessing with LangChain agent...\n")
            response = agent.query(user_input)
            print(f"Response:\n{response}\n")
            print("-" * 50)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
