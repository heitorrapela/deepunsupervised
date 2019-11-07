import re

def write_output(som_model, output_path, result):
    
    output_file = open(output_path, 'w+')

    n_clusters = som_model.node_control[som_model.node_control == 1].size(0)

    content = str(n_clusters) + "\t" + str(som_model.input_size) + "\n"
    for i, relevance in enumerate(som_model.relevance):
        if som_model.node_control[i] == 1:
            content += str(i) + "\t" + "\t".join(map(str, relevance.numpy())) + "\n"

    result_text = result.to_string(header=False, index=False).strip()
    result_text = re.sub('\n +', '\n', result_text)
    result_text = re.sub(' +', '\t', result_text)

    content += result_text
    output_file.write(content)
    output_file.close()