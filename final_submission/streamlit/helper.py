import pandas as pd

# Plotting libraries
import plotly.express as px
import plotly.graph_objects as go

# Google API libraries
import google.generativeai as palm
import base64
import json

# -------------------------------------------------------------------------------- #
# Data Load
# -------------------------------------------------------------------------------- #

def load_data(excel_filepath, nrows=None):

    if nrows== None:
        data = pd.read_excel(excel_filepath)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
    else:
        data = pd.read_csv(excel_filepath, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)

    return data

# -------------------------------------------------------------------------------- #
# Plotting Functions
# -------------------------------------------------------------------------------- #


def plot_barchart(df):
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df.non_compliant, name="non_compliant"))
    fig.add_trace(go.Bar(x=df.index, y=df.compliant, name="compliant"))
    fig.add_trace(go.Line(x=df.index, y=df.total, name="total"))

    fig.data[0].marker.color = tuple(['darkblue'] * len(df))
    fig.data[1].marker.color = tuple(['lightblue'] * len(df))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=500) #paper_bgcolor='rgba(0,0,0,0)'

    return fig


def plot_barh(df):
    fig = go.Figure()
    trace1= px.bar(x=df.breach, y=df.index, orientation = 'h', title=df.columns[0])

    fig.add_trace(trace1.data[0])

    fig.update_layout(
        autosize=False,
        width=1000,
        height=500) #paper_bgcolor='rgba(0,0,0,0)'

    return fig

# -------------------------------------------------------------------------------- #
# PALM Model
# -------------------------------------------------------------------------------- #

def config_palm():
    # Configure the client library by providing your API key.
    palm.configure(api_key="AIzaSyATNaHZH-ZC4yZ7HMpA13VLOmvSrSBIXHE")


def setup_palm():
  # These parameters for the model call can be set by URL parameters.
  model = 'models/text-bison-001' # @param {isTemplate: true}
  temperature = 0.75 # @param {isTemplate: true}
  candidate_count = 1 # @param {isTemplate: true}
  top_k = 40 # @param {isTemplate: true}
  top_p = 0.95 # @param {isTemplate: true}
  max_output_tokens = 1024 # @param {isTemplate: true}
  text_b64 = 'Q2xhc3NpZnkgdGhlIHNvY2lhbCBtZWRpYSBwb3N0IHRvIHdoZXRoZXIgaXQgaXMgY29tcGxpYW50IG9yIG5vbi1jb21wbGlhbnQsIGdpdmUgcmVhc29uIHRvIHdoeSBpdCBpcyBpbmNvbXBsaWFudCBhbmQgcHJvdmlkZSBzdWdnZXN0aW9ucyB0byBpbXByb3ZlIG9uIHRoZSBwb3N0IHNvIHRoYXQgaXQgaXMgY29tcGxpYW50LgoKQSBwb3N0IGlzIGluY29tcGxpYW50IHdoZW4gaXQgY29udGFpbnM6CigxKSBJdCBjb250YWlucyBhIG1pc3JlcHJlc2VudGF0aW9uIG9mIHRoZSBkZXNpZ25hdGlvbiBvZiB0aGUgcGVyc29uIHBvc3RpbmcuIEl0IHNob3VsZCBub3QgY29udGFpbiAnRmluYW5jaWFsIENvbnN1bHRhbnQnIC8gJ0ZpbmFuY2lhbCBBZHZpc29yJyBpbiB0aGUgdGV4dCAvIHRoZSBoYXN0YWdzLiAKKDIpIEl0IGlzIGEgbWlzcmVwcmVzZW50YXRpb24gb2YgdGhlIGNvbXBhbnkncyBwcm9maWxlLiBFZy4gZXhhZ2dlcmF0ZWQgdmFsdWUgb2YgcG90ZW50aWFsIGVhcm5pbmdzIGluIHRoZSBqb2IgcG9zdC4KKDMpIEl0IGlzIGEgbWlzcmVwcmVzZW50YXRpb24gb2YgdGhlIGNvbXBhbnkncyBwcm9kdWN0cy4gRWcuIGV4YWdnZXJhdGVkIHZhbHVlcyBpbiB0aGUgaW5zdXJhbmNlIHByb2R1Y3QgY292ZXJhZ2Ugb3IgcHJlbWl1bXMgb3IgZGlzY291bnRzLgooMykgSXQgaXMgYSBydWRlIHBvc3QuCig0KSBJdCBjb250YWlucyBjb250YWN0IGluZm9ybWF0aW9uIHRoYXQgYXJlIG5vdCBvZmZpY2lhbCBzdWNoIGFzIGdtYWlsLiBBbnkgZW1haWwgbm90IGVuZGluZyB3aXRoICdAcHJ1YWR2aXNlci5jb20uc2cnIGFyZSBub24tb2ZmaWNpYWwgZW1haWxzLgoKUG9zdDogSWYgeW91J3JlIHJlYWR5IHRvIGtpY2tzdGFydCB5b3VyIGNhcmVlciB3aXRoIGFuIGVhcm5pbmcgcG90ZW50aWFsIG9mICQyLjVrIC0gJDMuNWsgaW4geW91ciBmaXJzdCB5ZWFyLCB3ZSdkIGxvdmUgdG8gaGVhciBmcm9tIHlvdSEgQ29udGFjdCB1cyBhdCBwcnVmY0BnbWFpbC5jb20gdG8gbGVhcm4gbW9yZSBhYm91dCB0aGlzIGV4Y2l0aW5nIG9wcG9ydHVuaXR5IGFuZCBob3cgeW91IGNhbiBiZWNvbWUgYSB2YWx1ZWQgbWVtYmVyIG9mIG91ciBkeW5hbWljIHRlYW0uIExldCdzIGJ1aWxkIGEgcHJvc3Blcm91cyBmdXR1cmUgdG9nZXRoZXIhIPCfkrzwn5KqCsKgI0ZpbmFuY2lhbENvbnN1bHRhbnRPcHBvcnR1bml0ecKgI0pvaW5PdXJUZWFtwqAjRWFybmluZ1BvdGVudGlhbMKgI1Byb2Zlc3Npb25hbEdyb3d0aMKgI0ZpbmFuY2lhbFN1Y2Nlc3MKCkNvbmNsdXNpb246IFRoaXMgaXMgYW4gaW5jb21wbGlhbnQgcG9zdC4KUmVhc29uczogVGhlIHBvc3QgaXMgY29uc2lkZXJlZCBpbmNvbXBsaWFudCBiZWNhdXNlIGl0IG1heSBwb3RlbnRpYWxseSBtaXNyZXByZXNlbnQgUHJ1ZGVudGlhbCdzIHByb2ZpbGUgd2l0aCB0aGUgc3RhdGVkIGVhcm5pbmcgcG90ZW50aWFsIG9mICQyLjVrIC0gJDMuNWsgaW4gdGhlIGZpcnN0IHllYXIuIFRoaXMgZWFybmluZyByYW5nZSBzZWVtcyBxdWl0ZSBoaWdoIGFuZCBtaWdodCBnaXZlIGEgZmFsc2UgaW1wcmVzc2lvbiB0byBwb3RlbnRpYWwgY2FuZGlkYXRlcy4KU3VnZ2VzdGlvbnM6IEhlcmUgYXJlIHNvbWUgaW1wcm92ZW1lbnRzIHRvIG1ha2UgdGhlIHBvc3QgY29tcGxpYW50IGFuZCBtb3JlIHRyYW5zcGFyZW50OgoxLiBSZW1vdmUgU3BlY2lmaWMgRWFybmluZ3M6IEluc3RlYWQgb2YgcHJvdmlkaW5nIGEgc3BlY2lmaWMgZWFybmluZyBwb3RlbnRpYWwsIHlvdSBjYW4gdXNlIG1vcmUgZ2VuZXJhbCBsYW5ndWFnZSB0byBkZXNjcmliZSB0aGUgb3Bwb3J0dW5pdHkuIEZvciBleGFtcGxlOgoiSWYgeW91J3JlIHJlYWR5IHRvIGtpY2tzdGFydCB5b3VyIGNhcmVlciB3aXRoIGNvbXBldGl0aXZlIGVhcm5pbmdzIGluIHlvdXIgZmlyc3QgeWVhciwgd2UnZCBsb3ZlIHRvIGhlYXIgZnJvbSB5b3UhIgoyLiBQcm92aWRlIEFkZGl0aW9uYWwgRGV0YWlsczogVG8gZW5zdXJlIHRyYW5zcGFyZW5jeSwgeW91IGNhbiBtZW50aW9uIHRoYXQgZWFybmluZ3Mgd2lsbCB2YXJ5IGJhc2VkIG9uIGZhY3RvcnMgbGlrZSBwZXJmb3JtYW5jZSwgbG9jYXRpb24sIGFuZCB0aGUgcm9sZS4gRm9yIGV4YW1wbGU6CiJFYXJuaW5ncyB3aWxsIHZhcnkgYmFzZWQgb24geW91ciBwZXJmb3JtYW5jZSwgbG9jYXRpb24sIGFuZCB0aGUgcm9sZSwgYnV0IHdlIG9mZmVyIGNvbXBldGl0aXZlIGNvbXBlbnNhdGlvbi4iCjMuIEVtcGhhc2l6ZSBDYXJlZXIgR3Jvd3RoOiBJbnN0ZWFkIG9mIGZvY3VzaW5nIHNvbGVseSBvbiBlYXJuaW5ncywgaGlnaGxpZ2h0IHRoZSBwb3RlbnRpYWwgZm9yIGNhcmVlciBncm93dGggYW5kIGRldmVsb3BtZW50IHdpdGhpbiB0aGUgY29tcGFueS4gRm9yIGV4YW1wbGU6CiJKb2luIHVzIHRvIGVtYmFyayBvbiBhIHJld2FyZGluZyBjYXJlZXIgam91cm5leSB3aXRoIG9wcG9ydHVuaXRpZXMgZm9yIHByb2Zlc3Npb25hbCBncm93dGguIgo0LiBFbmNvdXJhZ2UgSW5xdWlyaWVzOiBLZWVwIHRoZSBpbnZpdGF0aW9uIHRvIGNvbnRhY3QgeW91IGZvciBtb3JlIGluZm9ybWF0aW9uLCBidXQgcmVtb3ZlIGFueSBzcGVjaWZpYyBtZW50aW9uIG9mIGVhcm5pbmdzLiBCdXQgZG8gcHJvdmlkZSBhIG9mZmljaWFsIGNoYW5uZWwgb2YgY29tbXVuaWNhdGlvbnMgaWUuIHBydWFkdmlzZXIgZW1haWxzIG9ubHkuCkJ5IG1ha2luZyB0aGVzZSBjaGFuZ2VzLCB0aGUgcG9zdCB3aWxsIGJlIG1vcmUgY29tcGxpYW50LCB0cmFuc3BhcmVudCwgYW5kIGxlc3MgbGlrZWx5IHRvIGdpdmUgcG90ZW50aWFsIGNhbmRpZGF0ZXMgZmFsc2UgZXhwZWN0YXRpb25zIGFib3V0IHRoZWlyIGVhcm5pbmdzIHBvdGVudGlhbC4KClBvc3Q6Cg==' # @param {isTemplate: true}
  stop_sequences_b64 = 'W10=' # @param {isTemplate: true}
  safety_settings_b64 = 'W3siY2F0ZWdvcnkiOiJIQVJNX0NBVEVHT1JZX0RFUk9HQVRPUlkiLCJ0aHJlc2hvbGQiOjF9LHsiY2F0ZWdvcnkiOiJIQVJNX0NBVEVHT1JZX1RPWElDSVRZIiwidGhyZXNob2xkIjoxfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9WSU9MRU5DRSIsInRocmVzaG9sZCI6Mn0seyJjYXRlZ29yeSI6IkhBUk1fQ0FURUdPUllfU0VYVUFMIiwidGhyZXNob2xkIjoyfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9NRURJQ0FMIiwidGhyZXNob2xkIjoyfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9EQU5HRVJPVVMiLCJ0aHJlc2hvbGQiOjJ9XQ==' # @param {isTemplate: true}

  # Convert the prompt text param from a bae64 string to a string.
  text = base64.b64decode(text_b64).decode("utf-8")

  # Convert the stop_sequences and safety_settings params from base64 strings to lists.
  stop_sequences = json.loads(base64.b64decode(stop_sequences_b64).decode("utf-8"))
  safety_settings = json.loads(base64.b64decode(safety_settings_b64).decode("utf-8"))

  defaults = {
    'model': model,
    'temperature': temperature,
    'candidate_count': candidate_count,
    'top_k': top_k,
    'top_p': top_p,
    'max_output_tokens': max_output_tokens,
    'stop_sequences': stop_sequences
  }

  return text, defaults