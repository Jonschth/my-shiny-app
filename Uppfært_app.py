import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os
from shiny import App, ui, render
import uvicorn
import networkx as nx


heildarkerfi_texti = "Hér er teknar saman 10 fyrirtækjasamstæður í aflahlutdeildarkerfinu miðað við heildarþorskígildistonn innan þess kerfis"
krókaflskerfi_texti = "Hér eru teknar saman 10 fyrirtækjasamsstæðurn í krókaaflakerfinu miðað við heildarþorskígildistonn innnan þess kerfis"
eignarhaldstalfa_texti = "Í þessari töflu koma fram innbyrðis eignatengsl sem notuð eru til þess að reikna yfirráð"
breytingar_texti = "Hér er unnt að velja samsetningu af nokkrum tilvikum. Auk þess má færa inn áhrif af kaupum og sameiningum í greininni"
samthoppun_texti = "Í töflunni hér að neðan eru reiknaðir helstu samþjöppunarstuðlar miðað við þau tilvik sem verið er að skoða hverju sinni"
eignatengsla_texti = "Hér birtast örvarit sem sýna tengsl á milli fyrirtækja innan samstæða miðað við þau tilvik sem hafa verið valin"

#import nest_asyncio #breyting 1
#nest_asyncio.apply() # breyting 2

def tengjast(gagnasafn):
  '''
  Fall til að tengjast gagnagrunnunum, Fiskur.db er gagnagrunnurinn með stærstu 100
  fyrirækjunum í sjávarútveginum og hlutfall þeirra af heildar ÞÍG sem og fisktegununum
  Args:
      gagnasafn (str): slóðin í gagnagrunninn
  '''
  conn = sqlite3.connect(gagnasafn)

  df = pd.read_sql('SELECT * FROM tafla', conn)
#   type(df)
  conn.close()
  if gagnasafn.split('/')[-1] == 'Fiskur23_24_uppfært_makríll.db':
    # print(df)
    if 'index' in df.columns:
        df = df.set_index('index').rename_axis('Eigendur', axis = 'index')
    elif 'Eigandi' in df.columns:
        df = df.set_index('Eigandi').rename_axis('Eigendur', axis = 'index')
  elif gagnasafn.split('/')[-1] == 'beta.db':
    df = df.set_index('Fyrirtæki').rename_axis('Fyriræki', axis = 'index')
  elif gagnasafn.split('/')[-1] == 'Krókur1.db':
    df = df.set_index('Eigandi')

  return df

def position(ÞÍG_per_fyrirtæki, hlutdeild_fylki, percentage = 20, fisktegund = '%-ÞÍG', mhhi = False, tréð = False, best = False, nuverandi = False):
  '''
  Adjusts the ÞÍG of each company and drops those that are subsidiaries of anohter
  Args:
      ÞÍG_per_fyrirtæki (DataFrame): Market share vector
      hlutdeild_fylki (DataFrame): ownership matrix of the 100 biggest fishing companies in Iceland
      percentage (Integer): Ownership control percentage
      fisktegund (String): column index
  '''
  if mhhi or nuverandi:
    percentage = 50
     
  dótturfélög = {fyrirtæki: [] for fyrirtæki in hlutdeild_fylki.index}

  for dótturfélag in hlutdeild_fylki.iterrows():
    for móðurfyrirtæki in hlutdeild_fylki.items():
      if dótturfélag[0] != móðurfyrirtæki[0]:
        if hlutdeild_fylki.loc[dótturfélag[0],móðurfyrirtæki[0]] >= percentage/100:
            dótturfélög[móðurfyrirtæki[0]].append(dótturfélag[0])

  hlutdeild_í_dótturfélögum = {fyrirtæki: [hlutdeild_fylki.loc[eign, fyrirtæki] for eign in eignir]
                             for fyrirtæki, eignir in dótturfélög.items()}
  if tréð:
    #  print('hér')
     return dótturfélög, hlutdeild_í_dótturfélögum
  stopper = {fyrirtæki: False for fyrirtæki in dótturfélög}

  droppa = []
  samstæður = []



  hlutdeild_fylki = hlutdeild_fylki - np.identity(hlutdeild_fylki.shape[0])
  for fyrirtæki in ÞÍG_per_fyrirtæki.iterrows():
    hlutdeild_fylki = _recursive_(dótturfélög, hlutdeild_fylki, fyrirtæki[0], hlutdeild_í_dótturfélögum, fisktegund, stopper, droppa, samstæður)
  # Vera viss að það fór ekkert á milli
  for column in hlutdeild_fylki.columns:
    for index, value in hlutdeild_fylki[column].items():
        if 0.5 < value < 1:
            hlutdeild_fylki.at[index, column] = 1.0
            droppa.append(index)
            samstæður.append(column)
  if mhhi or nuverandi:
    hlutdeild_fylki2 = hlutdeild_fylki.applymap(lambda x: 0 if 0 <= x < 0.5 else x)
  else:
    hlutdeild_fylki2 = hlutdeild_fylki.applymap(lambda x: 0 if 0 <= x < 0.2 else x)
  hlutdeild_fylki2 = hlutdeild_fylki2 + np.identity(hlutdeild_fylki.shape[0])
#   if fisktegund == '%-ÞÍG':
#     print(ÞÍG_per_fyrirtæki)
#     print(hlutdeild_fylki2)
#   print(hlutdeild_fylki2)

  ÞÍG_per_fyrirtæki = (ÞÍG_per_fyrirtæki.T@hlutdeild_fylki2).T
  if best:
     return ÞÍG_per_fyrirtæki
  ÞÍG_per_fyrirtæki = ÞÍG_per_fyrirtæki.drop(droppa)
  samstæður = [fyrirtæki for fyrirtæki in samstæður if fyrirtæki not in droppa]
  samstæður = list(set(samstæður))
  
  ÞÍG_per_fyrirtæki.rename(index={fyrirtæki: f'{fyrirtæki} samst.' for fyrirtæki in samstæður}, inplace=True)
  ÞÍG_per_fyrirtæki.rename(index={'Samherji Ísland ehf.': 'Samherji samst.'}, inplace=True)
  if mhhi or best:
    hlutdeild_fylki = hlutdeild_fylki.drop(droppa, axis = 0)
    hlutdeild_fylki = hlutdeild_fylki.drop(droppa, axis = 1)
    hlutdeild_fylki.rename(index={fyrirtæki: f'{fyrirtæki} samst.' for fyrirtæki in samstæður}, inplace=True)
    hlutdeild_fylki.rename(columns={fyrirtæki: f'{fyrirtæki} samst.' for fyrirtæki in samstæður}, inplace=True)
    hlutdeild_fylki.rename(index={'Samherji Ísland ehf.': 'Samherji samst.'}, inplace=True)
    hlutdeild_fylki.rename(columns={'Samherji Ísland hf.': 'Samherji samst.'}, inplace=True)

  return ÞÍG_per_fyrirtæki, hlutdeild_fylki

def _recursive_(dótturfélög, hlutdeild_fylki, fyrirtæki, hlutdeild_í_dótturfélögum, fisktegund, stopper, droppa, samstæður):
    try:   
      if len(dótturfélög[fyrirtæki]) != 0:
        for n, félag in enumerate(dótturfélög[fyrirtæki]):
    
          if stopper[fyrirtæki]:
            return hlutdeild_fylki
    
    
          elif stopper[félag]:
            if hlutdeild_í_dótturfélögum[fyrirtæki][n] > 0.5:
              hlutdeild_fylki.loc[:, fyrirtæki] = hlutdeild_fylki.loc[:,fyrirtæki] \
              + hlutdeild_fylki.loc[:, félag]
              hlutdeild_fylki.at[félag,fyrirtæki] = 1.0
              droppa.append(félag)
              samstæður.append(fyrirtæki)
    
            else:
              hlutdeild_fylki.loc[:,fyrirtæki] = hlutdeild_fylki.loc[:,fyrirtæki] \
              + hlutdeild_fylki.loc[:, félag]*hlutdeild_í_dótturfélögum[fyrirtæki][n]
    
          else:
            hlutdeild_fylki = _recursive_(dótturfélög, hlutdeild_fylki, félag, hlutdeild_í_dótturfélögum, fisktegund, stopper, droppa, samstæður)
            if hlutdeild_í_dótturfélögum[fyrirtæki][n] > 0.5:
              hlutdeild_fylki.loc[:,fyrirtæki] = hlutdeild_fylki.loc[:,fyrirtæki] \
              + hlutdeild_fylki.loc[:, félag]
              hlutdeild_fylki.at[félag,fyrirtæki] = 1.0
              droppa.append(félag)
              samstæður.append(fyrirtæki)
    
            else:
              hlutdeild_fylki.loc[:,fyrirtæki] = hlutdeild_fylki.loc[:,fyrirtæki] \
              + hlutdeild_fylki.loc[:, félag]*hlutdeild_í_dótturfélögum[fyrirtæki][n]
    
      stopper[fyrirtæki] = True
    
    except KeyError as e:
        print ('I got a KeyError - reason "%s"' % str(e))
    return hlutdeild_fylki

def calculate_mhhi_delta(s,beta):
    """Calculate mhhi_delta.

    Args:
        s (DataFrame): Market share vector
        beta (DataFrame): ownership matrix of the 100 biggest fishing companies in Iceland

    Returns
    -------
        DataFrame: with mhhi delta values
    """

    beta = beta.astype(float) 

    #Spegla
    beta = beta + beta.T
    # print(beta[:,1]@beta[])

    sum3 = 0

    for telja1, j in enumerate(beta.T):
        for telja2, k in enumerate(beta.T):
            if telja1 != telja2:
                sum4 = j @ k.T
                if sum4 != 0:
                    sum5 = j @ j.T
                    
                    # if telja1 == 1 or telja2 == 1:
                    #     print(telja1, telja2)
                    #     print(f'Samherji: sum 4: {sum4} sum 5: {sum5}, samtals: {sum4/sum5}')
                    #     print()
                    # if telja2 == 7 or telja2 == 7:
                    #     print(telja1, telja2)
                    #     print(f'Brim: sum 4: {sum4} sum 5: {sum5}, samtals: {sum4/sum5}')
                    #     print()
                    
                    # if sum4/sum5 > 100:
                    # #    print()
                    # #    print(telja1, telja2, sum4/sum5)
                    # #    print()
            
                    sum3 += s[telja1] * s[telja2] * 10000 * sum4 / sum5

    return sum3

def calculate_hhi(s):
      """Calculate hhi.

      Args:
          s (DataFrame): Market share vector

      Returns
      -------
          DataFrame: with ordinary hhi calculation
      """
      numpy_utgafa = np.sum(np.power(s,2), axis = 0)*10000
      return numpy_utgafa

def calculate_mhhi(df_fiskur, beta):

   hhi = calculate_hhi(df_fiskur.to_numpy())
   fylki = beta.to_numpy()
   fylki[fylki <= 0.01] = 0
   delta = calculate_mhhi_delta(df_fiskur.to_numpy(), fylki)

   A = hhi + delta

   A = pd.DataFrame(A.T, columns=['MHHI gildi'])

   A['Tegund'] = df_fiskur.columns
   A = A[['Tegund','MHHI gildi']]

   return A

def crn(n, fiskur):
   nump = fiskur.to_numpy()
   d = nump[0:n,:].sum(axis = 0)
   return d


level = [1]

def tré(fyrirtæki, dótturfélög,hlutdeild_í_dótturfélögum):
    '''
    Setur upp eignarhaldstréð

    fyrirtæki (String): Hvaða fyrirtæki er verið að taka fyrir
    dótturfélög (dict): dict með dótturfélögum félaga
    hlutdeild_í_dótturfélögum (DataFrame): Beta skjalið
    
    '''
   
    G = nx.DiGraph()
    # print(f'fyrirtæki: {str(fyrirtæki)}')

    def _endurkvæmt_(fyrirtæki, dótturfélög,hlutdeild_í_dótturfélögum):
        if dótturfélög[fyrirtæki]:
            for i, dótturfélag in enumerate(dótturfélög[fyrirtæki]):
                G.add_edge(fyrirtæki, dótturfélag, ownership = f'{hlutdeild_í_dótturfélögum[fyrirtæki][i]*100:.2f}%')
                if dótturfélög[dótturfélag]:
                    _endurkvæmt_(dótturfélag, dótturfélög,hlutdeild_í_dótturfélögum)

    _endurkvæmt_(fyrirtæki, dótturfélög,hlutdeild_í_dótturfélögum)

    def staða_í_grafi(graph, root):
        global level
        level = [1]
        levels = {root: 0}
        positions = {root: (0, 0)}
        queue = [root]

        while queue:
            node = queue.pop(0)
            level.append(levels[node])
            neighbors = list(graph.successors(node))
            width = len(neighbors)
            for i, child in enumerate(neighbors):
                levels[child] = level[-1] + 1
                positions[child] = (-width/3 + i, -level[-1]-1) # Breyting 5
                queue.append(child)

        return positions
    try:
        pos = staða_í_grafi(G, fyrirtæki)
    except:
        global level
        level = [1]
        G.add_node(fyrirtæki)
        pos = nx.spring_layout(G)
    colors = ["red" if node == fyrirtæki else "skyblue" for node in G.nodes()]
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=colors, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'ownership')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    ax.set_title(f"Eignatré {fyrirtæki}")
    # plt.gca().set_constrained_layout(True)
    # plt.show()



#path = "r:/Ráðgjöf/SVN2/Grein í Vísbendingu/gagnaskrár/"
#dir = "C:/Users\JST\Downloads\Apps\App for MHHI/"
dir=""
path = os.path.dirname(__file__)


prenta = True
plott = True
hve_mikið_prenta = 10
df_fiskur = tengjast(os.path.join(path,dir,'sfs_test.db'))
df_fiskur.rename(index={'Hraðfrystihús Hellissands hf':'Hraðfrystihús Hellissands hf.'},inplace=True) #22_23
df_fiskur.rename(index={'Steinunn hf.':'Steinunn ehf.'},inplace=True)#22_23


df_fiskur.set_index("index", inplace=True)

#df_fiskur = df_fiskur[~df_fiskur.index.duplicated(keep='first')]

df_beta = tengjast(os.path.join(path,dir, 'beta.db'))
#df_beta.rename(columns={'Vísir ehf.':'Vísir hf.'},inplace=True) #22_23
#df_beta.rename(index={'Vísir ehf.':'Vísir hf.'},inplace=True) #22_23

listi = df_fiskur.index
df_beta.drop(df_beta.index[~df_beta.index.isin(listi)], inplace=True)
df_beta=df_beta[df_beta.columns.intersection(listi)]


listi = df_beta.index
df_fiskur.drop(df_fiskur.index[~df_fiskur.index.isin(listi)], inplace=True)


df_krókur = tengjast(os.path.join(path,dir,'Krókur1.db'))
#df_krókur.drop(df_krókur.index[~df_krókur.index.isin(listi)], inplace=True)

# hér er búin til dictionary fyrir öll félög og dótturfélög þeirra sett í lista
dótturfélög = {fyrirtæki: [] for fyrirtæki in df_beta.index}

for dótturfélag in df_beta.iterrows():
    for móðurfyrirtæki in df_beta.items():
        if dótturfélag[0] != móðurfyrirtæki[0]:
            if df_beta.loc[dótturfélag[0],móðurfyrirtæki[0]] >= 20/100:
                dótturfélög[móðurfyrirtæki[0]].append(dótturfélag[0])

    hlutdeild_í_dótturfélögum = {fyrirtæki: [df_beta.loc[eign, fyrirtæki] for eign in eignir]
                                for fyrirtæki, eignir in dótturfélög.items()}


def bera_saman(df1,df2):
    diff = df1 != df2
    coordinates = [(row, col) for row, col in zip(*diff.to_numpy().nonzero())]
    coordinates = [(diff.index[row], diff.columns[col]) for row, col in coordinates]

    return coordinates


df_beta_krókur = df_beta

global notast
notast = df_beta.copy()

for row in df_krókur.index:
  if row not in df_beta_krókur.index:

    röð = pd.DataFrame(np.zeros((1,df_beta_krókur.shape[1])))
    röð = röð.rename(index={0: row})
    röð.columns = df_beta_krókur.columns

    df_beta_krókur = pd.concat([df_beta_krókur, röð], axis=0)
    dálkur = pd.DataFrame(np.zeros((df_beta_krókur.shape[0],1)))
    dálkur.at[df_beta_krókur.shape[0]-1,0] = 1.0
    dálkur.index = df_beta_krókur.index
    dálkur = dálkur.rename(columns={0:row})
    df_beta_krókur = pd.concat([df_beta_krókur, dálkur], axis = 1)

for row in df_beta_krókur:
  if row not in df_krókur.index:
    df_krókur.loc[row,'ÞÍG %'] = 0.0
    df_krókur.loc[row,'Þorskur %'] = 0.0
    df_krókur.loc[row,'Ýsa %'] = 0.0

app_ui = ui.page_fluid(
   
    ui.navset_pill_list(
        ui.nav_panel("Heildarkerfið",
        ui.tags.br(),
        ui.tags.div(
            ui.output_text_verbatim("Heildarkerfi", placeholder=False),
            style="font-family: Arial; font-size: 12px;"
        ),
    ui.div(
    ui.input_select("top", "Tíund:", choices=["00-10","11-20","21-30","31-40","41-50","51-60","61-70"]),
    style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;"
    ),

    ui.navset_card_tab(
    ui.nav_panel("Heildar",ui.output_plot("plot1"),ui.tags.br(), ui.tags.div(
        ui.output_table("data_table1",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

    ui.nav_panel("Karfinn", ui.output_plot("plot2"),ui.tags.br(),ui.tags.div(
        ui.output_table("data_table2",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

    ui.nav_panel("Þorskur", ui.output_plot("plot3"),ui.tags.br(),ui.tags.div(
        ui.output_table("data_table3",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

    ui.nav_panel("Ýsa", ui.output_plot("plot4"),ui.tags.br(),ui.tags.div(
        ui.output_table("data_table4",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

    ui.nav_panel("Ufsi", ui.output_plot("plot5"),ui.tags.br(),ui.tags.div(
        ui.output_table("data_table5",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

    ui.nav_panel("Grálúða", ui.output_plot("plot6"),ui.tags.br(),ui.tags.div(
        ui.output_table("data_table6",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

    ui.nav_panel("Síld", ui.output_plot("plot7"),ui.tags.br(),ui.tags.div(
        ui.output_table("data_table7",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

    ui.nav_panel("Úthafsrækja", ui.output_plot("plot8"),ui.tags.br(),ui.tags.div(
        ui.output_table("data_table8",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),
    )),

    ui.nav_panel("Krókaaflakerfið",
    ui.tags.br(),
    ui.output_text_verbatim("Krokaflskerfi", placeholder=False),
    ui.div(
    ui.input_select("top_krokur", "Tíund:", choices=["00-10","11-20","21-30","31-40","41-50","51-60","61-70"]),
    style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;"
    ),
    ui.navset_card_tab(
        ui.nav_panel('ÞÍG %', ui.output_plot("plot9",), ui.tags.br(),ui.tags.div(
        ui.output_table("data_table9",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

        ui.nav_panel('Þorskur %', ui.output_plot("plot10"), ui.tags.br(),ui.tags.div(
        ui.output_table("data_table10",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),

        ui.nav_panel('Ýsa %', ui.output_plot("plot11"), ui.tags.br(),ui.tags.div(
        ui.output_table("data_table11",style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
        style="text-align: center; margin: auto;" )),
    )),
    ui.nav_panel("Eignarhaldstaflan",
    ui.tags.br(),
    ui.output_text_verbatim("Eignarhaldstafla", placeholder=False),

    ui.output_table("data_table", style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: right;")),

    ui.nav_panel("Breytingar",
    ui.tags.br(),
    ui.output_text_verbatim("Breytingarnar", placeholder=False),

    ui.tags.hr(),

    ui.div(
    ui.input_checkbox("ÚR_KG", "Útgerðarfélag Reykjavíkur og KG Fiskverkun sameinað", value=False),
    ui.input_checkbox("ÚR_Brim", "Útgerðarfélag Reykjavíkur og Brim sameinað", value=False),
    ui.input_checkbox("Sam_SVN", "Samherji og Síldarvinnslan sameinað", value=False),
    ui.input_checkbox("Gamla", "Núverandi kerfi", value=False),

    style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;"
    ),
    ui.tags.hr(),
    
    ui.div(
        ui.input_select("select_column", "Eigendur:", choices=list(df_beta_krókur.columns)),
        ui.input_select("select_row", "Dótturfélag:", choices=list(df_beta_krókur.index)),
        ui.input_text("new_value", "Eignarhlutur eiganda í dótturfélagi, %:"),
        ui.input_action_button("submit", "Keyra",class_="btn-success"),
        style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;"
    ),
    ui.div(
    ui.output_text_verbatim("error_message"),
         style="color: red; font-weight: bold;"
    ),
    ui.tags.hr(),

    ui.output_text("munar","Breytingar gerðar")
    
    
    ),

    ui.nav_panel("Samþjöppun",
    ui.tags.br(),
    ui.output_text_verbatim("Samthoppun", placeholder=False),
    # ui.output_table("MHHI_gildi", style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),
    ui.output_table("HHI_gildi", style="margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"),   
        ) ,
    ui.nav_panel("Eignatengsl",
    # ui.tags.hr(),
    ui.tags.br(),
    ui.output_text_verbatim("Eignatengslin", placeholder=False),

    ui.input_select("select_company", "Fyrirtæki:", choices=list(df_beta_krókur.columns)),
    ui.div(
    {"style": "font-weight: bold;"},
    ui.output_plot("trémynd", height = "725px"),
    #style='width = "400px" height = "1600px"'
        )
    ),
    ui.tags.hr(),  # Horizontal line for separation
    

)ui.tags.div(
    ui.output_text("status_message"),
    style="font-weight: bold; text-align: center; font-size: 14px; color: red;"
),)
     

df_fiskur_global = df_fiskur
df_beta_global = df_beta
df_krókur_global = df_krókur
df_beta_krókur_global = df_beta_krókur

A = []
for column in df_fiskur_global.columns:
    s = df_fiskur_global[column].to_frame()
    s,hlutdeild = position(s, df_beta_global, 20, column)#, best = True)
    A.append(s)
sameinað_global = pd.concat(A, axis=1)
# print(hlutdeild)
A = []
for column in df_krókur_global.columns:
    s = df_krókur_global[column].to_frame()

    s,_ = position(s, df_beta_krókur_global, 20, column)#, best = True)
    A.append(s)
sameinað_krókur_global = pd.concat(A, axis=1)




upphaflega = 0
a,b,c,d = 0,0,0,0
e = 0
keyra = False
def server(input, output, session):

    def gogn(input, mhhi = False, tréð = False):
        global sameinað_global
        global sameinað_krókur_global
        global df_beta_global
        global df_beta_krókur_global
        global upphaflega
        global a,b,c,d,e
        global keyra

        if int(input.submit()) == 0:
            global gamla
            gamla = 0
        gamla_kerfið = input.Gamla()
        if gamla_kerfið:
           milli = 1
        else: milli = 0
        if not mhhi:

            # Binary til að sjá ef búið að breytast
            a = pow(2, 0) if input.ÚR_KG() else 0
            b = pow(2, 1) if input.ÚR_Brim() else 0
            c = pow(2, 2) if input.Sam_SVN() else 0
            d = pow(2, 3) if int(input.submit()) > gamla else 0
        if abs(e-milli)>0:
           e=milli
           keyra = True
        else: keyra = False        
        if mhhi or abs(upphaflega - (a+b+c+d)) > 0 or keyra:
            upphaflega = a+b+c+d

            if input.ÚR_KG():
                df_beta_global.at['KG Fiskverkun ehf.','Útgerðarfélag Reykjavíkur hf.'] = 0.51
                df_beta_krókur_global.at['KG Fiskverkun ehf.','Útgerðarfélag Reykjavíkur hf.'] = 0.51
            else:
                df_beta_global.at['KG Fiskverkun ehf.','Útgerðarfélag Reykjavíkur hf.'] = 0.0
                df_beta_krókur_global.at['KG Fiskverkun ehf.','Útgerðarfélag Reykjavíkur hf.'] = 0.0

            if input.ÚR_Brim():
                df_beta_global.at['Brim hf.','Útgerðarfélag Reykjavíkur hf.'] = 0.51
                df_beta_krókur_global.at['Brim hf.','Útgerðarfélag Reykjavíkur hf.'] = 0.51
            else:
                df_beta_global.at['Brim hf.','Útgerðarfélag Reykjavíkur hf.'] = 0.4396
                df_beta_krókur_global.at['Brim hf.','Útgerðarfélag Reykjavíkur hf.'] = 0.4396

            if input.Sam_SVN():
                df_beta_global.at['Síldarvinnslan hf.','Samherji Ísland ehf.'] = 0.51
                df_beta_krókur_global.at['Síldarvinnslan hf.','Samherji Ísland ehf.'] = 0.51
            else:
                df_beta_global.at['Síldarvinnslan hf.','Samherji Ísland ehf.'] = (30.06+0.15*3.49)/100
                df_beta_krókur_global.at['Síldarvinnslan hf.','Samherji Ísland ehf.'] = (30.06+0.15*3.49)/100


            if int(input.submit())> gamla:
                selected_column = input.select_column()
                selected_row = input.select_row()
                new_value = input.new_value()
                if selected_column and new_value is not None:
                    if selected_column == selected_row:
                        @render.text
                        def error_message():
                            return f"Þú valdir sama fyrirtækið í báðu"
                    elif df_beta_krókur_global.at[selected_column, selected_row] != 0:
                        @render.text
                        def error_message():
                            return f"Eignarhald fram og tilbaka"
                    
                    else:
                        
                        try:
                            new_value = float(new_value)/100
                            
                        
                            if new_value > 1.0 or new_value < 0.0:
                                @render.text
                                def error_message():
                                    return f"Gildið stærra en 1 eða minna en 0"
                            elif not ((float((df_beta_krókur_global-np.identity(df_beta_krókur_global.shape[0])).loc[selected_row,:].sum()) \
                                            -float(df_beta_krókur_global.at[selected_row,selected_column])+ float(new_value)) <= 1.0):
                                @render.text
                                def error_message():
                                    return f"Heildareign fyrirtækisins meiri en 100%"
                            else:              
                                try:
                                    df_beta_global.at[selected_row, selected_column] = float(new_value)
                                    df_beta_krókur_global.at[selected_row, selected_column] = float(new_value)
                                    @render.text
                                    def error_message():
                                        return f""
                                except ValueError:
                                    @render.text
                                    def error_message():
                                        return f"Villa"
                        except:
                            @render.text
                            def error_message():
                                return f"Ekki gild tala"
            
            gamla = int(input.submit())

            if tréð:
            #    print('mjá')
               return position(df_fiskur_global.iloc[:,0].to_frame(), df_beta_global, 20, '%-ÞÍG', tréð = True)
            nuverandi = False
            if gamla_kerfið:
                nuverandi = True
            @output
            @render.text
            def status_message():
                return "Þú ert núverandi kerfi" if nuverandi else "Þú ert í nýju lögunum"
            A = []
            df_beta2 = df_beta_global
            if gamla_kerfið:
               df_beta_global.at['Jakob Valgeir ehf.','Salting ehf.'] = 0.0
               df_beta_krókur_global.at['Jakob Valgeir ehf.','Salting ehf.'] = 0.0
            else: 
                df_beta_global.at['Jakob Valgeir ehf.','Salting ehf.'] = 0.75
                df_beta_krókur_global.at['Jakob Valgeir ehf.','Salting ehf.'] = 0.75

            pd.options.display.max_columns = None
            pd.options.display.max_rows = None
            for column in df_fiskur_global.columns:
                s = df_fiskur_global[column].to_frame()
                if mhhi:
                    s_global, df_beta2 = position(s, df_beta_global,20, column, mhhi)
                else:
                    s_global,abcd = position(s, df_beta_global, 20, column, nuverandi)
                    # print(abcd.at['Grunnur ehf.','Ísfélag hf.'])
                    # print(abcd.at['Þórsberg ehf.','Ísfélag hf.'])
                    # print(abcd.at['Melnes ehf.','Ísfélag hf.'])
                A.append(s_global)
            if mhhi:
               sameinað = pd.concat(A, axis = 1)
            else:
                sameinað_global = pd.concat(A, axis=1)#.sort_values(by='%-ÞÍG',ascending=False)
            A = []
            df_beta_krókur2 = df_beta_krókur_global
            for column in df_krókur.columns:
                s = df_krókur_global[column].to_frame()
                if mhhi:
                    s_global, df_beta_krókur2 = position(s, df_beta_krókur_global, 20, column, mhhi)
                else:
                    s_global,_ = position(s, df_beta_krókur_global, 20, column, nuverandi)
                A.append(s_global)
            if mhhi:
               sameinað_krókur = pd.concat(A, axis = 1)
               return sameinað.mul(100), df_beta2, sameinað_krókur.mul(100), df_beta_krókur2
            else:
                sameinað_krókur_global = pd.concat(A, axis=1)


            return sameinað_global.mul(100), df_beta2, sameinað_krókur_global.mul(100), df_beta_krókur2
        else:
           if tréð:
            #   print('hæhæ')
              return position(df_fiskur_global.iloc[:,0].to_frame(), df_beta_global, 20, '%-ÞÍG', tréð = True)
           return sameinað_global.mul(100), df_beta_global, sameinað_krókur_global.mul(100), df_beta_krókur_global

    @render.plot
    def trémynd():
        #if input.select_company:
        global level
        dótturfélög, hlutdeild_í_dótturfélögum=gogn(input, tréð = True)
        # print(dótturfélög)
        tré(input.select_company(), dótturfélög, hlutdeild_í_dótturfélögum)

    # @render.table
    # def MHHI_gildi():
    #     df_fiskur,df_beta,_,_ = gogn(input, mhhi = True)
    #     a = calculate_mhhi(df_fiskur.mul(1/100),df_beta)
    #     return a#.set_index('Tegund')
#    A['Tegund'] = df_fiskur.columns
#    A = A[['Tegund','MHHI gildi']]

    @render.table
    def HHI_gildi():
        df_fiskur,df_beta,_,_ = gogn(input, mhhi = True)
        hhi = calculate_hhi(df_fiskur.mul(1/100))
        mhhi = calculate_mhhi(df_fiskur.mul(1/100),df_beta)
        cr3 = crn(3,df_fiskur)
        cr8 = crn(8,df_fiskur)
        c3 = pd.DataFrame(cr3)
        c8 = pd.DataFrame(cr8)
        pandas_utgafa = pd.DataFrame(hhi)
        # mhhi.insert(1,"HHI gildi", pandas_utgafa)
        pandas_utgafa = pandas_utgafa.reset_index(drop=True)
        mhhi = mhhi.assign(HHI_gildi=pandas_utgafa, Cr3_prósenta = c3, Cr8_prósenta = c8)
        


        return mhhi#.set_index('Tegund')
    @render.text
    def munar():
       Listi=[]
       if input.Gamla():
          notast.at['Jakob Valgeir ehf.','Salting ehf.'] = 0.0
       else: notast.at['Jakob Valgeir ehf.','Salting ehf.'] = 0.75
       _,beta_skjal,_,_ = gogn(input)
       coordinates = bera_saman(notast,beta_skjal)
       if coordinates:
        return str("\n".join([f'...............................{tveir} á {beta_skjal.loc[einn,tveir]*100}% í {einn}' for einn, tveir in coordinates]))
       else: return ""


    @render.text
    def Samthoppun():
       global samthoppun_texti
       return samthoppun_texti
    @render.text
    def Heildarkerfi():
       global heildarkerfi_texti
       return heildarkerfi_texti
    @render.text
    def Krokaflskerfi():
       global krókaflskerfi_texti
       return krókaflskerfi_texti
    @render.text
    def Eignarhaldstafla():
       global eignarhaldstalfa_texti
       return eignarhaldstalfa_texti
    @render.text
    def Breytingarnar():
       global breytingar_texti
       return breytingar_texti
    @render.text
    def Eignatengslin():
       global eignatengsla_texti
       return eignatengsla_texti
   

    @render.table
    def data_table():
       _, _, _, df_beta_krókur = gogn(input)
       
       df_beta_krókur = df_beta_krókur - np.identity(df_beta_krókur.shape[0])
       df_beta_krókur = df_beta_krókur.loc[:, (df_beta_krókur != 0).any(axis=0)]

       df_beta_krókur = df_beta_krókur.loc[(df_beta_krókur != 0).any(axis=1)]

       return df_beta_krókur.mul(100).reset_index().rename(columns={'index': 'Fyrirtæki'})

    @render.table
    def data_table1():
       hopur = input.top()
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='%-ÞÍG',ascending=False)

       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table2():
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Karfi/gullkarfi',ascending=False)
       hopur = input.top()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table3():
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Þorskur',ascending=False)
       hopur = input.top()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table4():
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Ýsa',ascending=False)
       hopur = input.top()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table5():
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Ufsi',ascending=False)
       hopur = input.top()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table6():
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Grálúða',ascending=False)
       hopur = input.top()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table7():
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Síld',ascending=False)
       hopur = input.top()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table8():
       gognin, _, _, _ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Úthafsrækja',ascending=False)
       hopur = input.top()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    
    @render.table
    def data_table9():
       _,_,gognin,_ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='ÞÍG %',ascending=False)
       hopur = input.top_krokur()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table10():
       _,_,gognin,_ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Þorskur %',ascending=False)
       hopur = input.top_krokur()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)
    @render.table
    def data_table11():
       _,_,gognin,_ = gogn(input)
       gognin = gognin.reset_index().rename(columns={'index': 'Fyrirtæki'}).sort_values(by='Ýsa %',ascending=False)
       hopur = input.top_krokur()
       return pd.concat([gognin.iloc[:,0:1][int(hopur[0:2]):int(hopur[3:5])],gognin.iloc[:,1:2][int(hopur[0:2]):int(hopur[3:5])]], axis = 1)

    @render.plot
    def plot1():
        sameinað, _, _, _ = gogn(input)  # Assuming this function returns the data
        hopur = input.top()  # Get the top value (index range like '00-10')
        dálkur = sameinað.columns[0]  # Assuming the first column is the one to plot
        
        # Sort the data by the column in descending order
        sameinað = sameinað.sort_values(by=dálkur, ascending=False)
        plt.style.use('ggplot')

        #sns.set_style("whitegrid")    
        # Extract the start and end indices from hopur (e.g., "00-10")
        start_idx = int(hopur[0:2])  # Start index
        end_idx = int(hopur[3:5])    # End index
    
        
        # Create the plot figure
        plt.figure(figsize=(10, 12))
    
        # Plot the bars
        bars = plt.bar(
            sameinað.index[start_idx:end_idx],  # Slice the index properly
            sameinað[dálkur].iloc[start_idx:end_idx],  # Use .iloc for row slicing
            color='#007acc', edgecolor='black', alpha=0.7, width=0.8  # Adjust width for better visibility
        )
    
        # Add the height values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            # Ensure that the text is above the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # Position the text at the center of the bar
                yval + 0.5,  # Offset the text slightly above the bar
                round(yval, 2), ha='center', va='bottom', fontsize=10, color='black'
            )
    
        # Add horizontal lines for reference
        plt.axhline(y=0.15*100, color='g', linestyle='-', label='Hámark kauphöll')
        plt.axhline(y=0.12*100, color='r', linestyle='--', label='Hámark einkaeign')
    
        # Adjust the Y-axis to make the bars more visible if the values are small
        #plt.ylim(0, sameinað[dálkur].iloc[start_idx:end_idx].max() * 1.2)  # Add 20% margin above the max value
        plt.ylim(0,18)
    
        new_labels = [label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) for label in sameinað.index[start_idx:end_idx]]
        plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
    
        # Add labels and title
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall ÞÍG %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=12, fontname='Arial', fontweight='bold')
    
        # Add legend
        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

    @render.plot
    def plot2():
        sameinað, _, _, _ = gogn(input)  # Ensure `gogn(input)` is correctly defined
        dálkur = 'Karfi/gullkarfi'
    
        # Sort the DataFrame based on 'Karfi/gullkarfi'
        sameinað = sameinað.sort_values(by=dálkur, ascending=False)
    
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 12))
    
        # Extracting range from `hopur`
        hopur = input.top()  # Ensure `input` is a valid object
        start_idx, end_idx = map(int, hopur.split('-'))  # Correct index slicing
    
        # Dynamically set y-limit
        plt.ylim(0, 35)
    
        # Plot bars
        bars = plt.bar(
            sameinað.index[start_idx:end_idx], 
            sameinað[dálkur].iloc[start_idx:end_idx], 
            color='#007acc', edgecolor='black', alpha=0.7
        )
    
        # Annotate bars with values
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
    
        # Red dashed line as threshold
        plt.axhline(y=35, color='r', linestyle='--', label=f'Hámark {dálkur}')  # 35% threshold
    
        # Adjust x-axis labels (wrap if too long)
        new_labels = [label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) for label in sameinað.index[start_idx:end_idx]]

        plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
    
        # Labels and title
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall Karfi/gullkarfi %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=12, fontname='Arial', fontweight='bold')
    
        # Add legend
        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

    @render.plot
    def plot3():
        sameinað, _, _, _ = gogn(input)  # Ensure `gogn(input)` is correctly defined
        dálkur = 'Þorskur'
    
        # Sort by 'Þorskur' in descending order
        sameinað = sameinað.sort_values(by=dálkur, ascending=False)
    
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 12))
    
        # Extract range from `hopur`
        hopur = input.top()  # Ensure `input.top()` returns the correct format
        start_idx, end_idx = map(int, hopur.split('-'))  # Safe index extraction
    
        # Dynamically adjust y-axis limit
        plt.ylim(0, sameinað[dálkur].iloc[start_idx:end_idx].max() + 2)
    
        # Plot bars
        bars = plt.bar(
            sameinað.index[start_idx:end_idx], 
            sameinað[dálkur].iloc[start_idx:end_idx], 
            color='#007acc', edgecolor='black', alpha=0.7
        )
    
        # Annotate bars with values
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
    
        # Red dashed threshold line
        plt.axhline(y=12, color='red', linestyle='--', label=f'Hámark {dálkur}')  # 12% threshold
    
        # Adjust x-axis labels (wrap if too long)
        new_labels = [
            label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) 
            for label in sameinað.index[start_idx:end_idx]
        ]
        plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
    
        # Labels and title
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall ÞÍG %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=14, fontname='Arial', fontweight='bold')
    
        # Add legend
        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)



    @render.plot
    def plot4():
        sameinað, _, _, _ = gogn(input)  # Ensure `gogn(input)` is correctly defined
        dálkur = 'Ýsa'
    
        # Sort DataFrame by 'Ýsa' in descending order
        sameinað = sameinað.sort_values(by=dálkur, ascending=False)
    
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 12))
    
        # Extracting range from `hopur`
        hopur = input.top()  # Ensure `input` provides the correct format
        start_idx, end_idx = map(int, hopur.split('-'))  # Safe index extraction
    
        # Dynamically adjust y-axis limit
        plt.ylim(0, 20)
    
        # Plot bars
        bars = plt.bar(
            sameinað.index[start_idx:end_idx], 
            sameinað[dálkur].iloc[start_idx:end_idx], 
            color='#007acc', edgecolor='black', alpha=0.7
        )
    
        # Annotate bars with values
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
    
        # Red dashed line as threshold
        plt.axhline(y=20, color='r', linestyle='--', label=f'Hámark {dálkur}')  # 20% threshold
    
        # Adjust x-axis labels (wrap if too long)
        new_labels = [
            label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) 
            for label in sameinað.index[start_idx:end_idx]
        ]
        plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
    
        # Labels and title
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')  # Fixed typo
        plt.ylabel('Hlutfall Ýsa %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=14, fontname='Arial', fontweight='bold')
    
        # Add legend
        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

    @render.plot
    def plot5():
        sameinað, _,_,_ = gogn(input)
        dálkur = 'Ufsi'
        sameinað = sameinað.sort_values(by = dálkur, ascending = False)
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 12))

        hopur = input.top()
        bars = plt.bar(sameinað.index[int(hopur[0:2]):int(hopur[3:5])], sameinað[dálkur][int(hopur[0:2]):int(hopur[3:5])], color='#007acc', edgecolor='black', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')

        plt.axhline(y=0.2*100, color='r', linestyle='--', label=f'Hámark {dálkur}')

        plt.xticks(rotation=45, fontsize=10, fontname='Arial')
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall ÞÍG %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall tíu útgerðanna af heildarmarkaðinum', fontsize=14, fontname='Arial', fontweight='bold')


        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

    @render.plot
    def plot6():
        sameinað, _,_,_ = gogn(input)
        dálkur = 'Grálúða'
        sameinað = sameinað.sort_values(by = dálkur, ascending = False)
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 12))

        hopur = input.top()
        bars = plt.bar(sameinað.index[int(hopur[0:2]):int(hopur[3:5])], sameinað[dálkur][int(hopur[0:2]):int(hopur[3:5])], color='#007acc', edgecolor='black', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')

        plt.axhline(y=0.2*100, color='r', linestyle='--', label=f'Hámark {dálkur}')

        plt.xticks(rotation=45, fontsize=10, fontname='Arial')
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall ÞÍG %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=14, fontname='Arial', fontweight='bold')


        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

    @render.plot
    def plot7():
        sameinað, _, _, _ = gogn(input)  # Ensure `gogn(input)` is correctly defined
        dálkur = 'Síld'
    
        # Sort DataFrame by 'Síld' in descending order
        sameinað = sameinað.sort_values(by=dálkur, ascending=False)
    
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))
    
        # Extracting range from `hopur`
        hopur = input.top()  # Ensure `input` provides the correct format
        start_idx, end_idx = map(int, hopur.split('-'))  # Safe index extraction
    
        # Dynamically adjust y-axis limit
        plt.ylim(0, sameinað[dálkur].iloc[start_idx:end_idx].max() + 2)
    
        # Plot bars
        bars = plt.bar(
            sameinað.index[start_idx:end_idx], 
            sameinað[dálkur].iloc[start_idx:end_idx], 
            color='#007acc', edgecolor='black', alpha=0.7
        )
    
        # Annotate bars with values
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
    
        # Red dashed line as threshold
        plt.axhline(y=20, color='r', linestyle='--', label=f'Hámark {dálkur}')  # 20% threshold
    
        # Adjust x-axis labels (wrap if too long)
        new_labels = [
            label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) 
            for label in sameinað.index[start_idx:end_idx]
        ]
        plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
    
        # Labels and title
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall Síld %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=14, fontname='Arial', fontweight='bold')
    
        # Add legend
        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

    @render.plot
    def plot8():
        sameinað, _, _, _ = gogn(input)  # Assuming gogn is a function that returns the required dataframe and other values
        dálkur = 'Úthafsrækja'  # Column name for sorting
        
        # Ensure 'sameinað' is sorted by the column 'Úthafsrækja'
        sameinað = sameinað.sort_values(by=dálkur, ascending=False)
        
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 12))
    
        # Use top() to get the top values, assuming it is returning a series
        hopur = input.top()  # This is where you might need to adjust based on your data structure
        # Hopur is being used incorrectly for slicing, ensure the correct index values are being used
        start_idx = int(hopur[0:2])  # Adjust based on how hopur is structured
        end_idx = int(hopur[3:5])  # Adjust similarly
    
        # Use slicing on the DataFrame to plot bars correctly
        bars = plt.bar(sameinað.index[start_idx:end_idx], sameinað[dálkur].iloc[start_idx:end_idx], color='#007acc', edgecolor='black', alpha=0.7)
    
        # Annotating bars with their heights
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
    
        # Horizontal line at a specific y-value for the red line (could represent a threshold)
        plt.axhline(y=0.2 * 100, color='r', linestyle='--', label=f'Hámark {dálkur}')
    
        # Customize the ticks, labels, and title
        plt.xticks(rotation=45, fontsize=10, fontname='Arial')
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall ÞÍG %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=14, fontname='Arial', fontweight='bold')
    
        # Display the plot
        plt.legend()
        
    @render.plot
    def plot9():
        _, _, sameinað, _ = gogn(input)  # Assuming gogn is a function that returns the required dataframe and other values
        dálkur = 'ÞÍG %'  # Column name for sorting
        
        # Sort the DataFrame based on the column 'ÞÍG %'
        sameinað = sameinað.sort_values(by=dálkur, ascending=False)
        
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 12))
    
        # Set y-axis limits
        plt.ylim(0, 10)
        
        # Assuming 'hopur' is a slice/index to select the top rows, adjust it accordingly
        hopur = input.top_krokur()  # Adjust based on your data structure
        start_idx = int(hopur[0:2])  # Extract the start index (this may need tweaking)
        end_idx = int(hopur[3:5])  # Extract the end index (this may need tweaking)
    
        # Plot the bars
        bars = plt.bar(
            sameinað.index[start_idx:end_idx],
            sameinað[dálkur].iloc[start_idx:end_idx],
            color='#007acc', edgecolor='black', alpha=0.7
        )
    
        # Annotating bars with their heights
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                yval + 0.5, 
                round(yval, 2), 
                ha='center', va='bottom'
            )
    
        # Horizontal line at a specific y-value for the red line (could represent a threshold)
        plt.axhline(y=0.05 * 100, color='r', linestyle='--', label=f'Hámark {dálkur}')
        
        # Wrap xticks labels if they are too long
        new_labels = [label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) for label in sameinað.index[start_idx:end_idx]]
        plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
    
        # Customize the labels and title
        plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
        plt.ylabel('Hlutfall ÞÍG %', fontsize=10, fontname='Arial')
        plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=10, fontname='Arial', fontweight='bold')
    
        # Add legend and show the plot
        plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)
        


        @render.plot
        def plot10():
            _, _, sameinað, _ = gogn(input)  # Ensure `gogn(input)` returns correct data
            dálkur = 'Þorskur %'
        
            # Sort DataFrame based on 'Þorskur %'
            sameinað = sameinað.sort_values(by=dálkur, ascending=False)
        
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 12))
        
            # Extract range for filtering
            hopur = input.top_krokur()  # Ensure `input` is the correct object
            start_idx, end_idx = map(int, hopur.split('-'))
        
            # Adjust y-limit dynamically
            plt.ylim(0, sameinað[dálkur].iloc[start_idx:end_idx].max() + 2)
        
            # Plot bars
            bars = plt.bar(
                sameinað.index[start_idx:end_idx], 
                sameinað[dálkur].iloc[start_idx:end_idx], 
                color='#007acc', edgecolor='black', alpha=0.7
            )
        
            # Annotate bars with values
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
        
            # Red dashed line as threshold
            plt.axhline(y=4, color='r', linestyle='--', label=f'Hámark {dálkur}')  # 4% threshold
        
            # Adjust x-axis labels (wrap if too long)
            new_labels = [label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) for label in sameinað.index[start_idx:end_idx]]
            plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
        
            # Labels and title
            plt.xlabel('Nánar um útgerðirnar', fontsize=10, fontname='Arial')
            plt.ylabel('Hlutfall Þorskur %', fontsize=10, fontname='Arial')
            plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=10, fontname='Arial', fontweight='bold')
        
            # Add legend
            plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

        @render.plot
        def plot11():
            _, _, sameinað, _ = gogn(input)  # Ensure gogn(input) is correctly defined
            dálkur = 'Ýsa %'
        
            # Sort the DataFrame based on 'Ýsa %'
            sameinað = sameinað.sort_values(by=dálkur, ascending=False)
        
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 12))
        
            # Extracting range from `hopur`
            hopur = input.top_krokur()  # Ensure `input` is a valid object
            start_idx, end_idx = map(int, hopur.split('-'))  # Correct index slicing
        
            # Dynamically set y-limit
            plt.ylim(0, sameinað[dálkur].iloc[start_idx:end_idx].max() + 2)
        
            # Plot bars
            bars = plt.bar(
                sameinað.index[start_idx:end_idx], 
                sameinað[dálkur].iloc[start_idx:end_idx], 
                color='#007acc', edgecolor='black', alpha=0.7
            )
        
            # Annotate bars with values
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
        
            # Red dashed line as threshold
            plt.axhline(y=5, color='r', linestyle='--', label=f'Hámark {dálkur}')  # 5% threshold
        
            # Adjust x-axis labels (wrap if too long)
            new_labels = [label if len(label) <= 15 else '\n'.join([label[i:i+15] for i in range(0, len(label), 15)]) for label in sameinað.index[start_idx:end_idx]]
            plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45, ha='right', fontsize=10, fontname='Arial')
        
            # Labels and title
            plt.xlabel('Nánar um útgerðirnar', fontsize=12, fontname='Arial')
            plt.ylabel('Hlutfall Ýsa %', fontsize=12, fontname='Arial')
            plt.title('Hlutfall útgerðanna af heildarmarkaðinum', fontsize=14, fontname='Arial', fontweight='bold')
        
            # Add legend
            plt.legend(loc='upper right', framealpha=1, fontsize=10, frameon=True)

app = App(app_ui, server)

port = int(os.getenv("PORT", 8000))  # Render assigns the PORT environment variable
uvicorn.run(app, host="0.0.0.0", port=port)