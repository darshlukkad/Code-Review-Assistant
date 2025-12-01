"""
Streamlit frontend for AI Code Review Assistant.

Beautiful, interactive web interface for code quality analysis.
"""

import streamlit as st
import requests
from typing import Dict
import json


# Page configuration
st.set_page_config(
    page_title="AI Code Review Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .issue-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .issue-critical {
        background-color: #fee;
        border-color: #dc2626;
    }
    
    .issue-high {
        background-color: #fef3c7;
        border-color: #f59e0b;
    }
    
    .issue-medium {
        background-color: #dbeafe;
        border-color: #3b82f6;
    }
    
    .issue-low {
        background-color: #f0fdf4;
        border-color: #10b981;
    }
    
    .quality-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# API Configuration
API_URL = st.sidebar.text_input(
    "API Endpoint",
    value="http://localhost:8000",
    help="FastAPI backend URL"
)


def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """
    Call backend API.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data
    
    Returns:
        Response dictionary
    """
    try:
        url = f"{API_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def display_header():
    """Display app header."""
    st.markdown('<h1 class="main-header">🔍 AI Code Review Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by CodeBERT • Detect bugs, security issues, and code smells instantly</p>', unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with options."""
    st.sidebar.title("⚙️ Settings")
    
    # Detection threshold
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score to report an issue"
    )
    
    # Language selection
    language = st.sidebar.selectbox(
        "Programming Language",
        options=["python", "javascript"],
        help="Select the programming language of your code"
    )
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.title("ℹ️ About")
    st.sidebar.info(
        "This tool uses a fine-tuned CodeBERT model to detect:\n\n"
        "🐛 **Bugs** - Runtime errors and logical issues\n\n"
        "🔒 **Security** - Vulnerabilities and unsafe practices\n\n"
        "💨 **Performance** - Optimization opportunities\n\n"
        "✨ **Style** - Code quality and formatting\n\n"
        "🚨 **Code Smells** - Design issues"
    )
    
    # Example code
    st.sidebar.markdown("---")
    if st.sidebar.button("📝 Load Example Code"):
        st.session_state['example_loaded'] = True
    
    return threshold, language


def get_example_code() -> str:
    """Get example code to demonstrate the tool."""
    return """def calculate_average(numbers):
    # Calculate the average of a list of numbers
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: ZeroDivisionError if empty

def validate_password(password):
    # Security: Weak password validation
    if len(password) < 6:
        return False
    return True

def process_data(data):
    # Performance: Nested loops
    result = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            result.append(data[i][j] * 2)
    return result
"""


def display_quality_score(score: float):
    """
    Display overall quality score.
    
    Args:
        score: Quality score (0-100)
    """
    # Determine color based on score
    if score >= 80:
        color = "#10b981"  # Green
        emoji = "✅"
    elif score >= 60:
        color = "#3b82f6"  # Blue
        emoji = "ℹ️"
    elif score >= 40:
        color = "#f59e0b"  # Orange
        emoji = "⚠️"
    else:
        color = "#dc2626"  # Red
        emoji = "❌"
    
    st.markdown(
        f'<div class="quality-score" style="color: {color};">{emoji} {score:.1f}/100</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"<p style='text-align: center; color: {color}; font-weight: bold;'>Code Quality Score</p>", unsafe_allow_html=True)


def display_issue(issue: Dict):
    """
    Display a single issue.
    
    Args:
        issue: Issue dictionary
    """
    severity = issue['severity']
    issue_type = issue['type']
    confidence = issue['confidence']
    description = issue['description']
    
    # Severity emoji and color
    severity_config = {
        'critical': {'emoji': '🚨', 'class': 'issue-critical'},
        'high': {'emoji': '⚠️', 'class': 'issue-high'},
        'medium': {'emoji': 'ℹ️', 'class': 'issue-medium'},
        'low': {'emoji': '💡', 'class': 'issue-low'}
    }
    
    config = severity_config.get(severity, severity_config['medium'])
    
    st.markdown(
        f"""
        <div class="issue-card {config['class']}">
            <h4>{config['emoji']} {issue_type.upper().replace('_', ' ')} ({severity.upper()})</h4>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main application logic."""
    display_header()
    
    # Sidebar
    threshold, language = display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📝 Code Input")
        
        # Load example if requested
        default_code = ""
        if st.session_state.get('example_loaded', False):
            default_code = get_example_code()
            st.session_state['example_loaded'] = False
        
        code_input = st.text_area(
            "Paste your code here:",
            value=default_code,
            height=400,
            placeholder="Enter Python or JavaScript code to analyze..."
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Code", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if analyze_button and code_input.strip():
            with st.spinner("Analyzing code..."):
                # Call API
                result = call_api(
                    "/review",
                    method="POST",
                    data={
                        "code": code_input,
                        "language": language,
                        "threshold": threshold
                    }
                )
                
                if result:
                    # Display quality score
                    display_quality_score(result['overall_quality_score'])
                    
                    st.markdown("---")
                    
                    # Display metrics
                    st.metric("Issues Detected", result['num_issues'])
                    
                    st.markdown("---")
                    
                    # Display issues
                    if result['issues']:
                        st.subheader("🔍 Detected Issues")
                        for issue in result['issues']:
                            display_issue(issue)
                    else:
                        st.success("✅ No issues detected! Your code looks great!")
        
        elif analyze_button:
            st.warning("⚠️ Please enter some code to analyze.")
        else:
            st.info("👈 Paste your code and click 'Analyze Code' to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Built with ❤️ using CodeBERT and Streamlit | "
        "CS 5590 Final Project</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
