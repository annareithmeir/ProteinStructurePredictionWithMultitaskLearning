def countLocations(sequence, classification_mode,parts):
    rest=len(sequence)%parts
    onePart=len(sequence)/parts

    if(classification_mode==3):
        matrix = np.zeros((parts, classification_mode), dtype=int)
        occurences=pd.DataFrame(matrix,
                                columns=['C','H','E'],
                                index=[range(parts)])
    else:
        matrix = np.zeros((parts, classification_mode), dtype=int)
        occurences = pd.DataFrame(matrix,
                                  columns=[ 'H', 'E', 'I','S','T','B','G','-'],
                                  index=[range(parts)])
    for i in range(parts):
        if(i==parts-1 and rest!=0):
            for j in range(rest+onePart):
                occurences[sequence[onePart * i + j]][i] += 1
        else:
            for j in range(onePart):
                occurences[sequence[onePart*i+j]][i]+=1

    return occurences

def countLocations_dict(dict, classification_mode, parts):
    for i in range(len(dict)):
        if('occs' in vars()):
            occs+=countLocations(dict[dict.keys()[i]][1], classification_mode, parts)
        else:
            occs=(countLocations(dict[dict.keys()[i]][1], classification_mode, parts))
    plotPartitionEach(occs)
    return occs

def plotPartitionEach(occ):
    plt.figure()
    plt.suptitle('Locations of structures in '+str(occ.shape[0])+' parts')
    plt.ylabel('Occurences')
    cols=['green', 'cyan', 'orange','blue','red','yellow','pink','brown','black']


    for i in range(occ.shape[1]):
        tmp=occ.keys()[i]
        plt.subplot(1, occ.shape[1], i+1)
        plt.xlabel(tmp+'(max:'+str(occ[tmp].max())+' min:'+str(occ[tmp].min())+')')
        plt.ylim([0,np.max(occ.values)])
        plt.bar(range(occ.shape[0])+np.ones(occ.shape[0]),occ[tmp], color=cols[i], )

    plt.savefig('CountLocations.pdf')

def maskSequences(residues, structure):
    if(len(structure)!=len(residues)):
        raise ValueError('[ERROR] Structure and Residues have different length!')
    maskedResidues=''
    maskedStructures=''

    for i in range(len(structure)):
        if(structure[i]!='X' and structure[i]!='Y'):
            maskedResidues+=residues[i]
            maskedStructures+=structure[i]

    return maskedResidues, maskedStructures
