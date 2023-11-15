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
    palm.configure(api_key="")


def setup_palm():
  
    """
    This function is used for setting up PaLM api call.
    """
  
    # These parameters for the model call can be set by URL parameters.
    model = 'models/text-bison-001' # @param {isTemplate: true}
    temperature = 0.75 # @param {isTemplate: true}
    candidate_count = 1 # @param {isTemplate: true}
    top_k = 40 # @param {isTemplate: true}
    top_p = 0.95 # @param {isTemplate: true}
    max_output_tokens = 1024 # @param {isTemplate: true}
    text_b64 = 'Q2xhc3NpZnkgdGhlIHNvY2lhbCBtZWRpYSBwb3N0IHRvIHdoZXRoZXIgaXQgaXMgY29tcGxpYW50IG9yIG5vbi1jb21wbGlhbnQsIGdpdmUgcmVhc29uIHRvIHdoeSBpdCBpcyBpbmNvbXBsaWFudCBhbmQgcHJvdmlkZSBzdWdnZXN0aW9ucyB0byBpbXByb3ZlIG9uIHRoZSBwb3N0IHNvIHRoYXQgaXQgaXMgY29tcGxpYW50LgogICAgICAKICAgICAgICAgIEEgcG9zdCBpcyBpbmNvbXBsaWFudCB3aGVuIGl0IGNvbnRhaW5zOgogICAgICAgICAgKDEpIEl0IGNvbnRhaW5zIGEgbWlzcmVwcmVzZW50YXRpb24gb2YgdGhlIGRlc2lnbmF0aW9uIG9mIHRoZSBwZXJzb24gcG9zdGluZywgc3VjaCBhcyBiZWluZyBhIGZpbmFuY2lhbCBjb25zdWx0YW50LgogICAgICAgICAgKDMpIEl0IGlzIGEgbWlzcmVwcmVzZW50YXRpb24gb2YgdGhlIGNvbXBhbnkncyBwcm9maWxlLiBFZy4gZXhhZ2dlcmF0ZWQgdmFsdWUgb2YgcG90ZW50aWFsIGVhcm5pbmdzIGluIHRoZSBqb2IgcG9zdC4KICAgICAgICAgICg0KSBJdCBpcyBhIG1pc3JlcHJlc2VudGF0aW9uIG9mIHRoZSBjb21wYW55J3MgcHJvZHVjdHMuIEVnLiBleGFnZ2VyYXRlZCB2YWx1ZXMgaW4gdGhlIGluc3VyYW5jZSBwcm9kdWN0IGNvdmVyYWdlIG9yIHByZW1pdW1zIG9yIGRpc2NvdW50cy4KICAgICAgICAgICg1KSBJdCBpcyBhIHJ1ZGUgcG9zdC4KICAgICAgICAgICg2KSBJdCBjb250YWlucyBjb250YWN0IGluZm9ybWF0aW9uIHRoYXQgYXJlIG5vdCBvZmZpY2lhbCBzdWNoIGFzIGdtYWlsLiBBbnkgZW1haWwgbm90IGVuZGluZyB3aXRoICdAcHJ1YWR2aXNlci5jb20uc2cnIGFyZSBub24tb2ZmaWNpYWwgZW1haWxzLgoKICAgICAgICAgIE5vdGU6CiAgICAgICAgICAoMSkgUHJ1ZGVudGlhbCBpcyBhbiBpbnN1cmFuY2UgY29tcGFueSBvZmZlcmluZyBtYWlubHkgbGlmZSBpbnN1cmFuY2UuIAogICAgICAgICAgKDIpIENvbnRhaW5pbmcgd29yZHMgbGlrZSBjb25zdWx0YXRpb24gaXMgZmluZS4gUHJ1ZGVudGlhbCBkb2VzIG9mZmVyIGZyZWUgY29uc3VsdGF0aW9ucy4KCiAgICAgICAgICBQb3N0OiBJZiB5b3UncmUgcmVhZHkgdG8ga2lja3N0YXJ0IHlvdXIgY2FyZWVyIHdpdGggYW4gZWFybmluZyBwb3RlbnRpYWwgb2YgJDIuNWsgLSAkMy41ayBpbiB5b3VyIGZpcnN0IHllYXIsIHdlJ2QgbG92ZSB0byBoZWFyIGZyb20geW91ISBDb250YWN0IHVzIGF0IHBydWZjQGdtYWlsLmNvbSB0byBsZWFybiBtb3JlIGFib3V0IHRoaXMgZXhjaXRpbmcgb3Bwb3J0dW5pdHkgYW5kIGhvdyB5b3UgY2FuIGJlY29tZSBhIHZhbHVlZCBtZW1iZXIgb2Ygb3VyIGR5bmFtaWMgdGVhbS4gTGV0J3MgYnVpbGQgYSBwcm9zcGVyb3VzIGZ1dHVyZSB0b2dldGhlciEg8J+SvPCfkqoKICAgICAgICAgICNGaW5hbmNpYWxDb25zdWx0YW50T3Bwb3J0dW5pdHkgI0pvaW5PdXJUZWFtICNFYXJuaW5nUG90ZW50aWFsICNQcm9mZXNzaW9uYWxHcm93dGggI0ZpbmFuY2lhbFN1Y2Nlc3MKCiAgICAgICAgICBDb25jbHVzaW9uOiBUaGlzIGlzIGFuIGluY29tcGxpYW50IHBvc3QuCiAgICAgICAgICBSZWFzb25zOiBUaGUgcG9zdCBpcyBjb25zaWRlcmVkIGluY29tcGxpYW50IGJlY2F1c2UgaXQgbWF5IHBvdGVudGlhbGx5IG1pc3JlcHJlc2VudCBQcnVkZW50aWFsJ3MgcHJvZmlsZSB3aXRoIHRoZSBzdGF0ZWQgZWFybmluZyBwb3RlbnRpYWwgb2YgJDIuNWsgLSAkMy41ayBpbiB0aGUgZmlyc3QgeWVhci4gVGhpcyBlYXJuaW5nIHJhbmdlIHNlZW1zIHF1aXRlIGhpZ2ggYW5kIG1pZ2h0IGdpdmUgYSBmYWxzZSBpbXByZXNzaW9uIHRvIHBvdGVudGlhbCBjYW5kaWRhdGVzLgogICAgICAgICAgU3VnZ2VzdGlvbnM6IEhlcmUgYXJlIHNvbWUgaW1wcm92ZW1lbnRzIHRvIG1ha2UgdGhlIHBvc3QgY29tcGxpYW50IGFuZCBtb3JlIHRyYW5zcGFyZW50OgogICAgICAgICAgMS4gUmVtb3ZlIFNwZWNpZmljIEVhcm5pbmdzOiBJbnN0ZWFkIG9mIHByb3ZpZGluZyBhIHNwZWNpZmljIGVhcm5pbmcgcG90ZW50aWFsLCB5b3UgY2FuIHVzZSBtb3JlIGdlbmVyYWwgbGFuZ3VhZ2UgdG8gZGVzY3JpYmUgdGhlIG9wcG9ydHVuaXR5LiBGb3IgZXhhbXBsZToKICAgICAgICAgICJJZiB5b3UncmUgcmVhZHkgdG8ga2lja3N0YXJ0IHlvdXIgY2FyZWVyIHdpdGggY29tcGV0aXRpdmUgZWFybmluZ3MgaW4geW91ciBmaXJzdCB5ZWFyLCB3ZSdkIGxvdmUgdG8gaGVhciBmcm9tIHlvdSEiCiAgICAgICAgICAyLiBQcm92aWRlIEFkZGl0aW9uYWwgRGV0YWlsczogVG8gZW5zdXJlIHRyYW5zcGFyZW5jeSwgeW91IGNhbiBtZW50aW9uIHRoYXQgZWFybmluZ3Mgd2lsbCB2YXJ5IGJhc2VkIG9uIGZhY3RvcnMgbGlrZSBwZXJmb3JtYW5jZSwgbG9jYXRpb24sIGFuZCB0aGUgcm9sZS4gRm9yIGV4YW1wbGU6CiAgICAgICAgICAiRWFybmluZ3Mgd2lsbCB2YXJ5IGJhc2VkIG9uIHlvdXIgcGVyZm9ybWFuY2UsIGxvY2F0aW9uLCBhbmQgdGhlIHJvbGUsIGJ1dCB3ZSBvZmZlciBjb21wZXRpdGl2ZSBjb21wZW5zYXRpb24uIgogICAgICAgICAgMy4gRW1waGFzaXplIENhcmVlciBHcm93dGg6IEluc3RlYWQgb2YgZm9jdXNpbmcgc29sZWx5IG9uIGVhcm5pbmdzLCBoaWdobGlnaHQgdGhlIHBvdGVudGlhbCBmb3IgY2FyZWVyIGdyb3d0aCBhbmQgZGV2ZWxvcG1lbnQgd2l0aGluIHRoZSBjb21wYW55LiBGb3IgZXhhbXBsZToKICAgICAgICAgICJKb2luIHVzIHRvIGVtYmFyayBvbiBhIHJld2FyZGluZyBjYXJlZXIgam91cm5leSB3aXRoIG9wcG9ydHVuaXRpZXMgZm9yIHByb2Zlc3Npb25hbCBncm93dGguIgogICAgICAgICAgNC4gRW5jb3VyYWdlIElucXVpcmllczogS2VlcCB0aGUgaW52aXRhdGlvbiB0byBjb250YWN0IHlvdSBmb3IgbW9yZSBpbmZvcm1hdGlvbiwgYnV0IHJlbW92ZSBhbnkgc3BlY2lmaWMgbWVudGlvbiBvZiBlYXJuaW5ncy4gQnV0IGRvIHByb3ZpZGUgYSBvZmZpY2lhbCBjaGFubmVsIG9mIGNvbW11bmljYXRpb25zIGllLiBwcnVhZHZpc2VyIGVtYWlscyBvbmx5LgogICAgICAgICAgQnkgbWFraW5nIHRoZXNlIGNoYW5nZXMsIHRoZSBwb3N0IHdpbGwgYmUgbW9yZSBjb21wbGlhbnQsIHRyYW5zcGFyZW50LCBhbmQgbGVzcyBsaWtlbHkgdG8gZ2l2ZSBwb3RlbnRpYWwgY2FuZGlkYXRlcyBmYWxzZSBleHBlY3RhdGlvbnMgYWJvdXQgdGhlaXIgZWFybmluZ3MgcG90ZW50aWFsLgoKICAgICAgICAgIFBvc3Q6IERvbid0IHdhaXQgdW50aWwgaXQncyB0b28gbGF0ZSEgVGFrZSB0aGUgcHJvYWN0aXZlIHN0ZXAgdG9kYXkgYW5kIHByb3RlY3QgeW91ciBmYW1pbHkncyBmdXR1cmUuIENvbnRhY3QgdXMgYXQgcGxlYXNlYnV5QGZyb21tZS5jb20gZm9yIGEgZnJlZSBjb25zdWx0YXRpb24uIExldCdzIGVuc3VyZSB5b3VyIGxvdmVkIG9uZXMgYXJlIHNoaWVsZGVkIGZyb20gbGlmZSdzIHVuY2VydGFpbnRpZXMgYW5kIGhhdmUgYSBicmlnaHRlciB0b21vcnJvdyEgCgogICAgICAgICAgQ29uY2x1c2lvbjogVGhpcyBpcyBhbiBpbmNvbXBsaWFudCBwb3N0LiBSZWFzb25zOiBUaGUgcG9zdCBpcyBjb25zaWRlcmVkIGluY29tcGxpYW50IGJlY2F1c2UgaXQgY29udGFpbnMgYSBtaXNyZXByZXNlbnRhdGlvbiBvZiB0aGUgZGVzaWduYXRpb24gb2YgdGhlIHBlcnNvbiBwb3N0aW5nLiBJdCBhbHNvIGNvbnRhaW5zIGEgbm9uLW9mZmljaWFsIGVtYWlsIGFkZHJlc3MuIFN1Z2dlc3Rpb25zOiBIZXJlIGFyZSBzb21lIGltcHJvdmVtZW50cyB0byBtYWtlIHRoZSBwb3N0IGNvbXBsaWFudDoKICAgICAgICAgIDEuIFVzZSBhIG1vcmUgb2ZmaWNpYWwgZW1haWwgYWRkcmVzcywgc3VjaCBhcyAicHJ1YWR2aXNlckBwcnVhZHZpc2VyLmNvbS5zZyIuCiAgICAgICAgICAyLiBSZXBocmFzZSB0aGUgY2FsbCB0byBhY3Rpb24gdG8gYmUgbW9yZSBzcGVjaWZpYyBhbmQgYWN0aW9uYWJsZS4gRm9yIGV4YW1wbGU6ICJDb250YWN0IHVzIGZvciBhIGZyZWUgY29uc3VsdGF0aW9uIHRvIGxlYXJuIG1vcmUgYWJvdXQgaG93IHdlIGNhbiBoZWxwIHlvdSBwcm90ZWN0IHlvdXIgZmFtaWx5J3MgZnV0dXJlLiIgQnkgbWFraW5nIHRoZXNlIGNoYW5nZXMsIHRoZSBwb3N0IHdpbGwgYmUgbW9yZSBjb21wbGlhbnQgYW5kIHdpbGwgbm90IG1pc2xlYWQgcG90ZW50aWFsIGN1c3RvbWVycy4KICAgICAgICAgIAogICAgICAgICAgUG9zdDo=' # @param {isTemplate: true}
    stop_sequences_b64 = 'W10=' # @param {isTemplate: true}
    safety_settings_b64 = 'W3siY2F0ZWdvcnkiOiJIQVJNX0NBVEVHT1JZX0RFUk9HQVRPUlkiLCJ0aHJlc2hvbGQiOjF9LHsiY2F0ZWdvcnkiOiJIQVJNX0NBVEVHT1JZX1RPWElDSVRZIiwidGhyZXNob2xkIjoxfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9WSU9MRU5DRSIsInRocmVzaG9sZCI6Mn0seyJjYXRlZ29yeSI6IkhBUk1fQ0FURUdPUllfU0VYVUFMIiwidGhyZXNob2xkIjoyfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9NRURJQ0FMIiwidGhyZXNob2xkIjoyfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9EQU5HRVJPVVMiLCJ0aHJlc2hvbGQiOjJ9XQ==' # @param {isTemplate: true}

    # Convert the stop_sequences and safety_settings params from base64 strings to lists.
    text = base64.b64decode(text_b64).decode("utf-8")
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