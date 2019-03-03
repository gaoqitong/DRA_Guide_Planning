import networkx as nx
import pydot as pd

def plot_buchi(buchi,filename):
    filename = './'+ filename +'_dot.txt'
    pdfname = filename.replace('.txt','.pdf')
    dotfile = nx.drawing.nx_pydot.write_dot(buchi, filename)
    with open(filename,'r') as readdot:
        modified_dot = readdot.read().replace('\n','').replace('guard_formula','label')
    with open(filename,'w') as write_new_dot:
        write_new_dot.write(modified_dot)
    
    graph = pd.graph_from_dot_file(filename)
    graph = graph.pop()
    graph.write_pdf(pdfname)
    
    print 'Graph Successfully Plotted'