# -*- coding: utf-8 -*-
"""
Leptospira Classifier - Local Script with Gradio
"""

import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import shutil
import tempfile
import subprocess
import traceback
import gradio as gr

# ------------------------------------------------
# Setup paths (assume repo already cloned locally)
# ------------------------------------------------
cloned_repo_path = "./Modelo_Leptospira"

# Define paths for the directories and files within the cloned repository
model_directory_path = os.path.join(cloned_repo_path, "models")
protein_sequences_directory_path = os.path.join(cloned_repo_path, "multi_fastas")
# The specific genome file path will be within the data directory of the cloned repo
new_genome_file_path = os.path.join(cloned_repo_path, "data", "120_Brem_307.fas")
# The training data file path will be within the data directory of the cloned repo
training_file_path = os.path.join(cloned_repo_path, "data", "serogroup_averages - Sheet2 (1).csv") # Corrected filename

print(f"Cloned repository path: {cloned_repo_path}")
print(f"Model directory path: {model_directory_path}")
print(f"Protein sequences directory path: {protein_sequences_directory_path}")
print(f"New genome file path: {new_genome_file_path}")
print(f"Training file path: {training_file_path}")
# Display all rows
pd.set_option('display.max_rows', None)

# ------------------------------------------------
# Data Loading and Preprocessing Functions
# ------------------------------------------------

#These functions handle the loading, verification, and preprocessing of the data for both training (used here to get training columns) and prediction.

def preprocess_for_prediction(table):
    """
    Preprocess data for machine learning prediction.

    Args:
        table (pd.DataFrame): The input DataFrame for prediction.
    Returns:

        original_indices (list): List of original row indices from the input DataFrame.
    """
    print("Starting preprocessing for prediction...")
    # Drop the specified columns that are not features, ignoring errors if they don't exist
    columns_to_exclude = ['Strain', 'Serogroup', 'Species']
    processed_table = table.drop(columns=columns_to_exclude, errors='ignore').copy()
    print(f"  - Columns '{columns_to_exclude}' removed. Current shape: {processed_table.shape}")

    # Ensure all columns are numeric, coercing non-numeric values to NaN
    print("  - Converting non-numeric columns to numeric...")
    non_numeric_columns = processed_table.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric_columns:
        print(f"    - Non-numeric columns detected: {non_numeric_columns}")
        for col in non_numeric_columns:
             processed_table[col] = pd.to_numeric(processed_table[col], errors='coerce')
    print("  - Column conversion completed.")

    # Drop rows with any remaining missing values (NaNs)
    rows_before_dropna = processed_table.shape[0]
    processed_table = processed_table.dropna()
    rows_after_dropna = processed_table.shape[0]
    if rows_after_dropna < rows_before_dropna:
        print(f"  - Removed {rows_before_dropna - rows_after_dropna} rows with missing values.")
    print(f"  - Shape after removing missing values: {processed_table.shape}")

    # Drop the 'class' column if it exists, as it's not a feature for prediction
    if 'class' in processed_table.columns:
        print("  - 'class' column found. Removing...")
        processed_table = processed_table.drop(columns=['class'])
        print(f"  - 'class' column removed. Current shape: {processed_table.shape}")
    else:
        print("  - 'class' column not found. Proceeding.")

    # Debugging: Print columns before prediction to verify
    print("\n  - Columns in DataFrame before prediction:")
    print(processed_table.columns.tolist())
    print(f"  - Número de colunas antes de predição: {processed_table.shape[1]}")

    # Ensure the order of columns matches the training data features.
    # This is crucial for consistent prediction.
    # A robust approach requires saving and loading the list of training feature column names.
    # Assuming for now that the remaining columns are in the correct relative order.

    print("Preprocessing for prediction completed.")
    # Return the feature matrix and the list of original indices for the rows that were kept
    return processed_table, processed_table.index.tolist()

def parse_fasta_to_tuples(fasta_path):
    """
    Reads a .fasta file and returns a list of (header, sequence) tuples.

    Args:
        fasta_path (str): Path to the .fasta file.

    Returns:
        list: A list of tuples, where each tuple contains the sequence header
              (without the '>' symbol) and the sequence string.
    """
    sequences = []
    header = None
    seq = ""

    # Open and read the fasta file line by line
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # If a header and sequence were previously collected, store them
                if header and seq:
                    sequences.append((header, seq))
                # Start a new sequence entry with the current header
                header = line[1:]  # remove the ">" character
                seq = ""
            else:
                # Append non-header lines to the current sequence
                seq += line
        # Add the last sequence in the file after the loop finishes
        if header and seq:
            sequences.append((header, seq))

    return sequences

def build_sequence_dictionary(protein_sequences_directory_path):
    """
    Reads all .fasta files in a directory and builds a dictionary mapping
    serogroup names (derived from filenames) to lists of protein sequences.

    Args:
        protein_sequences_directory_path (str): Path to the directory containing
                                                the protein sequence .fasta files.

    Returns:
        dict: A dictionary where keys are serogroup names and values are lists
              of (header, sequence) tuples parsed from the .fasta files.
    """
    protein_sequences_by_serogroup = {}

    # Iterate through each file in the specified directory
    for filename in os.listdir(protein_sequences_directory_path):
        # Process files with common FASTA extensions
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            fasta_path = os.path.join(protein_sequences_directory_path, filename)

            # Define serogroup name from the filename (assuming name before the first "_" or extension)
            # Example: 'Ballum_proteins.fasta' -> 'Ballum'
            # Or 'Ballum.fasta' -> 'Ballum'
            # Use filename without extension as the key
            serogroup_name = os.path.splitext(filename)[0]


            # Parse the fasta file into a list of (header, sequence) tuples
            sequence_list = parse_fasta_to_tuples(fasta_path)

            # Add the parsed sequences to the dictionary under the serogroup name key
            protein_sequences_by_serogroup[serogroup_name] = sequence_list

            print(f"Processed {filename} ({len(sequence_list)} sequences)")

    return protein_sequences_by_serogroup

"""## BLAST and Prediction Pipeline Functions

These functions handle running BLAST, parsing its output, and making predictions.
"""

def run_tblastn(genome_archive_path, protein_sequences_by_serogroup, output_file_path):
    """
    Runs tblastn with the given genome archive against the provided protein sequences,
    grouped by serogroup, and saves the output to a specified file.

    Args:
        genome_archive_path (str): Path to the genome archive file (e.g., .fasta, .fna).
        protein_sequences_by_serogroup (dict): A dictionary where keys are serogroup names
                                              and values are lists of (header, sequence) tuples.
        output_file_path (str): Path to save the tblastn output.

    Returns:
        bool: True if tblastn ran successfully for all serogroups, False otherwise.
    """
    print(f"Starting tblastn for genome archive: {os.path.basename(genome_archive_path)}")
    print(f"  - Saving tblastn output to: {output_file_path}")

    # Create a temporary directory for BLAST database files
    blast_db_dir = tempfile.mkdtemp()
    # print(f"  - Created temporary BLAST database directory: {blast_db_dir}") # Removed debug print

    # Define paths for temporary files within the temporary directory
    genome_fasta = os.path.join(blast_db_dir, "genome.fasta")
    database_name = os.path.join(blast_db_dir, "genome_db")

    # Extract genome from archive (assuming it's a fasta file inside) or copy if already fasta
    try:
        if genome_archive_path.endswith('.gz'):
             # Use gunzip to decompress and save to the temporary fasta file
             subprocess.run(['gunzip', '-c', genome_archive_path], stdout=open(genome_fasta, 'w'), check=True)
             # print(f"  - Extracted genome from {os.path.basename(genome_archive_path)} to {genome_fasta}") # Removed debug print
        else:
            # Just copy the file if it's not compressed
            subprocess.run(['cp', genome_archive_path, genome_fasta], check=True)
            # print(f"  - Copied genome file {os.path.basename(genome_archive_path)} to {genome_fasta}") # Removed debug print

    except subprocess.CalledProcessError as e:
        print(f"  - Error extracting or copying genome from archive: {e}")
        shutil.rmtree(blast_db_dir) # Clean up the temporary directory
        return False
    except FileNotFoundError:
        print(f"  - Error: Could not find command to extract or copy genome file.")
        shutil.rmtree(blast_db_dir) # Clean up the temporary directory
        return False


    # Create BLAST database from the genome fasta file
    try:
        print("  - Creating BLAST database...")
        # Use subprocess.Popen with pipes to capture potential large stdout/stderr
        makeblastdb_process = subprocess.Popen([
            "makeblastdb",
            "-in", genome_fasta,
            "-dbtype", "nucl", # Nucleotide database type
            "-out", database_name # Output database name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = makeblastdb_process.communicate()
        if makeblastdb_process.returncode != 0:
             print(f"  - Error creating BLAST database: {stderr.decode()}")
             shutil.rmtree(blast_db_dir) # Clean up the temporary directory
             return False
        print("  - BLAST database created successfully.")
    except FileNotFoundError:
        print(f"  - Error: 'makeblastdb' command not found. Ensure NCBI BLAST+ is installed and in your PATH.")
        shutil.rmtree(blast_db_dir) # Clean up the temporary directory
        return False
    except Exception as e:
        print(f"  - An unexpected error occurred during BLAST database creation: {e}")
        shutil.rmtree(blast_db_dir) # Clean up the temporary directory
        return False


    # Run tblastn for each serogroup's protein sequences against the genome database
    success = True
    # Ensure the output file is empty before appending results from each serogroup
    try:
        with open(output_file_path, 'w') as f:
            pass # Create or clear the file by opening in write mode and closing
    except IOError as e:
        print(f"  - Error: Could not write to output file {output_file_path}: {e}")
        shutil.rmtree(blast_db_dir) # Clean up the temporary directory
        return False


    for serogroup_name, sequences in protein_sequences_by_serogroup.items():
        if not sequences:
            # print(f"  - Skipping serogroup {serogroup_name}: No sequences found.") # Removed debug print
            continue

        # Define temporary FASTA file path for the current serogroup's proteins within the DB directory
        serogroup_protein_fasta = os.path.join(blast_db_dir, f"{serogroup_name}_proteins.fasta")

        # Create temporary FASTA file for the current serogroup's protein sequences
        with open(serogroup_protein_fasta, "w") as f:
            # Iterate through the list of (header, sequence) tuples
            for item in sequences:
                # Ensure the item is in the expected tuple format
                if isinstance(item, tuple) and len(item) == 2:
                    header, sequence = item
                    # Append the serogroup name to the header for identification in BLAST output
                    # Take only the first part of the header if it contains spaces
                    cleaned_header = header.split()[0]
                    f.write(f">{cleaned_header}_{serogroup_name}\n{sequence}\n")
                else:
                    # print(f"  - Warning: Skipping unexpected item format in sequences for {serogroup_name}: {item}") # Removed debug print
                    success = False # Indicate an issue but try to continue

        # Run tblastn for the current serogroup's proteins against the genome database
        try:
            # Use 'a' mode to append tblastn results to the main output file
            with open(output_file_path, 'a') as outfile:
                 tblastn_process = subprocess.Popen([
                     "tblastn",
                     "-query", serogroup_protein_fasta, # Protein sequences for the current serogroup
                     "-db", database_name, # Genome database
                     "-outfmt", "6 qseqid sseqid pident length qlen", # Output format
                     #"-max_target_seqs", "1", # Output format: query seq id, subject seq id, percent identity, alignment length, query length
                 ], stdout=outfile, stderr=subprocess.PIPE) # Direct stdout to the output file and capture stderr
                 stdout, stderr = tblastn_process.communicate()

                 if tblastn_process.returncode != 0:
                      print(f"  - Error running tblastn for {serogroup_name}: {stderr.decode()}")
                      success = False # Mark as failed but continue with other serogroups

            # print(f"  - tblastn completed for serogroup {serogroup_name}.") # Removed debug print

        except FileNotFoundError:
            print(f"  - Error: 'tblastn' command not found. Ensure NCBI BLAST+ is installed and in your PATH.")
            success = False
            break # Cannot continue if tblastn command is missing
        except Exception as e:
            print(f"  - An unexpected error occurred during tblastn for {serogroup_name}: {e}")
            success = False # Mark as failed but continue with other serogroups

        # Clean up the temporary serogroup protein FASTA file after use
        os.remove(serogroup_protein_fasta)

    # Clean up the temporary BLAST database directory and its contents
    shutil.rmtree(blast_db_dir)
    # print(f"  - Cleaned up temporary BLAST database directory: {blast_db_dir}") # Removed debug print

    print("tblastn execution completed.")
    return success

# NOTE: The cell that calls this function (cell_id: Kr-w5BdKjVUC) will need to be re-run after this modification.

def predict_serogroup_from_genome(genome_archive_path, protein_sequences_by_serogroup, loaded_model, training_feature_columns):
    """
    Runs tblastn on a genome archive, processes the output into a feature vector
    with columns matching the training data, and predicts the serogroup using
    a loaded model.

    Args:
        genome_archive_path (str): Path to the genome archive file (e.g., .fasta, .fna).
        protein_sequences_by_serogroup (dict): A dictionary where keys are serogroup names
                                              and values are lists of (header, sequence) tuples.
        loaded_model: The pre-trained machine learning model.
        training_feature_columns (list): A list of column names used as features during training.


    Returns:
        pd.DataFrame or None: A DataFrame containing the prediction and probabilities,
                              or None if the process fails.
    """
    print(f"Starting prediction pipeline for genome: {os.path.basename(genome_archive_path)}")

    # Define a temporary file path for the combined tblastn output
    # We will still save to a file, but also print its content for debugging
    blast_output_file = "/tmp/blast_output.txt"

    # Run tblastn using the protein sequences grouped by serogroup against the genome
    run_tblastn_result = run_tblastn(genome_archive_path, protein_sequences_by_serogroup, blast_output_file)

    if not run_tblastn_result:
        print("  - tblastn execution failed. Aborting prediction.")
        return None

    # --- Adicionado para imprimir a saída do BLAST para depuração ---
    # Removed debug print of BLAST output file content
    # --------------------------------------------------------------------


    # Parse the combined BLAST output and create a feature vector DataFrame
    # Pass the protein_sequences_by_serogroup dictionary to the parsing function for mapping
    # Remove the loaded_scaler argument
    feature_vector_df = parse_blast_output_and_create_feature_vector(blast_output_file, protein_sequences_by_serogroup, training_feature_columns)

    if feature_vector_df is None:
        print("  - Failed to create feature vector from BLAST output. Aborting prediction.")
        return None

    # Display the feature vector DataFrame containing the mean BLAST results per serogroup
    #print(f"\n--- Mean BLAST Pident per Serogroup for {os.path.basename(genome_archive_path)} ---")
    #display(feature_vector_df)
    print("--------------------------------------------------------------------")


    # Preprocess the feature vector (removed scaling)
    print("  - Preparing feature vector for prediction...")
    try:
        # Ensure the feature vector has the same columns and order as the training data before prediction
        # The parsing function should handle this, but reindex is a safeguard.
        if not feature_vector_df.columns.equals(pd.Index(training_feature_columns)):
             print("  - Warning: Feature vector columns do not match training columns. Attempting reindex.")
             # Reindex to match training columns, filling missing ones with 0.0
             feature_vector_df = feature_vector_df.reindex(columns=training_feature_columns, fill_value=0.0)
             # print("  - Feature vector reindexed.") # Removed debug print

        # Use the feature vector directly without scaling
        X_pred = feature_vector_df.values
        print("  - Feature vector ready for prediction.")
    except Exception as e:
        print(f"  - Error preparing feature vector: {e}")
        print("  - Ensure the feature vector columns and order match the model's expected input.")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        return None

    # Make predictions using the loaded machine learning model
    # print("  - Making predictions...") # Removed debug print
    try:
        predictions = loaded_model.predict(X_pred)
        prediction_probabilities = loaded_model.predict_proba(X_pred)
        # print("  - Predictions made successfully.") # Removed debug print
    except Exception as e:
        print(f"  - Error making predictions: {e}")
        print("  - Ensure the feature vector shape matches the model's expected input.")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        return None

    # Create a DataFrame to store the prediction results and probabilities
    # Use the genome file name as identification for the prediction row
    genome_name = os.path.basename(genome_archive_path)
    predictions_df = pd.DataFrame({
        'Genome': [genome_name],
        'Predicted_Class': predictions,
    })

    # Add columns for prediction probabilities for each class
    # Ensure the class labels from the model are used for column names
    if hasattr(loaded_model, 'classes_'):
        for i, class_label in enumerate(loaded_model.classes_):
             predictions_df[f'Probability_Class_{class_label}'] = prediction_probabilities[:, i]
    else:
        print("  - Warning: Model does not have 'classes_' attribute. Probability columns might not be correctly labeled.")
        # Fallback to generic column names if class labels are not available
        for i in range(prediction_probabilities.shape[1]):
             predictions_df[f'Probability_Class_{i+1}'] = prediction_probabilities[:, i]

    print("Prediction pipeline completed.")
    # Display the resulting predictions DataFrame
    # Removed display of final prediction DataFrame from this function,
    # as the overall results are concatenated and displayed later.
    # display(predictions_df)

    return predictions_df

# Define the path to the directory containing protein sequences for BLAST
# This path is now set in the initial setup cell after cloning.
# protein_sequences_directory_path = "https://github.com/Spidey2004/Modelo_Leptospira/tree/Spidey2004-patch-1.0/multi_fastas" # Old path

# Initialize an empty dictionary to store sequences grouped by serogroup
protein_sequences_by_serogroup = {}

# Check if the specified path is a directory
# We are now using the local path from the cloned repository.
if not os.path.isdir(protein_sequences_directory_path):
    print(f"Error: Directory not found at {protein_sequences_directory_path}")
else:
    print(f"Reading protein sequences from directory: {protein_sequences_directory_path}")
    # Iterate through each file in the specified directory
    for filename in os.listdir(protein_sequences_directory_path):
        # Process files with common FASTA extensions
        if filename.endswith(".fasta") or filename.endswith(".fna") or filename.endswith(".fa"):
            # Extract the serogroup name from the filename (assuming filename is the serogroup name)
            # Example: 'Ballum.fasta' -> 'Ballum'
            serogroup_name = os.path.splitext(filename)[0]
            file_path = os.path.join(protein_sequences_directory_path, filename)
            print(f"  - Reading file: {filename} for serogroup: {serogroup_name}")

            # Initialize a list to hold (header, sequence) tuples for the current serogroup
            protein_sequences_by_serogroup[serogroup_name] = []

            current_sequence = ""
            current_header = ""
            try:
                # Read the sequences from the current FASTA file
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            # If there's a header and sequence from the previous entry, store it
                            if current_header and current_sequence:
                                # Store the previous sequence as a tuple (header, sequence)
                                protein_sequences_by_serogroup[serogroup_name].append((current_header, current_sequence))

                            # Start a new sequence entry
                            current_header = line[1:] # Remove the '>' character
                            current_sequence = ""
                        elif line:
                            # Append non-header lines to the current sequence
                            current_sequence += line

                    # Add the last sequence in the file after the loop finishes
                    if current_header and current_sequence:
                         protein_sequences_by_serogroup[serogroup_name].append((current_header, current_sequence))

            except Exception as e:
                print(f"  - Error reading file {filename}: {e}")
                continue # Continue to the next file even if one fails

    print(f"Loaded protein sequences for {len(protein_sequences_by_serogroup)} serogroups from directory.")
    # The dictionary 'protein_sequences_by_serogroup' now contains serogroup names as keys
    # and a list of (header, sequence) tuples as values.

def parse_blast_output_and_create_feature_vector(blast_output_file, protein_sequences_by_serogroup, training_feature_columns):
    print(f"\n Iniciando o processamento do arquivo BLAST: {os.path.basename(blast_output_file)}")

    try:
        blast_cols = ['qseqid', 'sseqid', 'pident', 'length', 'qlen']
        print(" Lendo o arquivo BLAST (formato outfmt 6: qseqid, sseqid, pident, length, qlen)...")
        blast_df = pd.read_csv(
            blast_output_file,
            sep='\t',
            header=None,
            names=blast_cols,
            dtype={'pident': float, 'length': int, 'qlen': int}
        )

        print(f" Dados carregados: {blast_df.shape[0]} linhas.")
        blast_df.fillna(0.0, inplace=True)

        if blast_df.empty:
            print(" Nenhum resultado encontrado no BLAST. Criando vetor de características zerado.")
            # Return a DataFrame with all training feature columns, filled with 0.0
            return pd.DataFrame([{col: 0.0 for col in training_feature_columns}])

        print(" Calculando o pident corrigido para cada linha: pident * (length / qlen)...")
        blast_df['corrected_pident'] = blast_df.apply(
            lambda row: row['pident'] * (row['length'] / row['qlen']) if row['qlen'] > 0 else 0.0,
            axis=1
        )
        print(f" pident corrigido calculado. Exemplo:\n{blast_df[['pident', 'length', 'qlen', 'corrected_pident']].head()}")

        print("\n Selecionando o melhor hit (maior corrected_pident) por proteína consultada (qseqid)...")
        best_hits_per_protein = blast_df.loc[
            blast_df.groupby('qseqid')['corrected_pident'].idxmax()
        ].copy()
        print(f" {len(best_hits_per_protein)} melhores hits selecionados.")

        print(" Extraindo o nome do sorogrupo a partir do qseqid (última parte após '_')...")
        best_hits_per_protein['serogroup_name_from_qseqid'] = best_hits_per_protein['qseqid'].apply(
            lambda x: x.split("_")[-1] if "_" in x else 'Unknown'
        )

        print(" Verificando quais sorogrupos extraídos existem nas colunas de treinamento...")
        unique_qseqid_serogroups = best_hits_per_protein['serogroup_name_from_qseqid'].unique().tolist()
        valid_serogroup_names_in_blast = [
            name for name in unique_qseqid_serogroups if name in training_feature_columns
        ]
        print(f" Sorogrupos válidos encontrados: {valid_serogroup_names_in_blast}")

        print(" Filtrando apenas os melhores hits que pertencem aos sorogrupos válidos...")
        filtered_best_hits = best_hits_per_protein[
            best_hits_per_protein['serogroup_name_from_qseqid'].isin(valid_serogroup_names_in_blast)
        ].copy()
        print(f" {filtered_best_hits.shape[0]} hits mantidos após o filtro.")

        print(" Agrupando por sorogrupo e calculando a MÉDIA do pident corrigido...")
        mean_corrected_pident_by_serogroup = (
            filtered_best_hits.groupby('serogroup_name_from_qseqid')['corrected_pident'].mean()
            if not filtered_best_hits.empty
            else pd.Series(dtype=float)
        )
        print(" Média por sorogrupo calculada:")
        print(mean_corrected_pident_by_serogroup)

        print("\n Iniciando vetor de características com zero para todos os sorogrupos esperados...")
        # Initialize with all training feature columns, set to 0.0
        feature_vector_data = {col: 0.0 for col in training_feature_columns}

        print(" Preenchendo o vetor de características com as médias calculadas por sorogrupo...")
        for serogroup, mean_value in mean_corrected_pident_by_serogroup.items():
            # Only fill if the serogroup is in the expected training columns and not 'max'
            if pd.notna(mean_value) and serogroup in feature_vector_data and serogroup != 'max':
                print(f"  - Atribuindo {mean_value:.2f} a '{serogroup}'")
                feature_vector_data[serogroup] = mean_value

        # Remove the 'max' calculation and assignment
        # print("\n📈 Calculando a feature especial 'max' (maior valor entre as médias por sorogrupo)...")
        # if 'max' in training_feature_columns:
        #     max_of_means = mean_corrected_pident_by_serogroup.max() if not mean_corrected_pident_by_serogroup.empty else 0.0
        #     feature_vector_data['max'] = max_of_means if pd.notna(max_of_means) else 0.0
        #     print(f"  - Valor de 'max': {feature_vector_data['max']:.2f}")


        print("\n Construindo o DataFrame final com o vetor de características para predição...")
        feature_vector_df = pd.DataFrame([feature_vector_data])
        # Ensure columns are in the correct order and fill missing with 0.0
        feature_vector_df = feature_vector_df.reindex(columns=training_feature_columns, fill_value=0.0)
        feature_vector_df = feature_vector_df.fillna(0.0)

        print(f" Vetor de características criado com sucesso. Dimensão: {feature_vector_df.shape}")
        print(" Colunas preenchidas:", feature_vector_df.columns.tolist())
        print(" Valores finais:")
        print(feature_vector_df)

        return feature_vector_df

    except FileNotFoundError:
        print(f" Erro: Arquivo não encontrado em {blast_output_file}")
        return None
    except Exception as e:
        print(" Erro inesperado durante o parsing:")
        traceback.print_exc()
        return None

def parse_fasta_to_tuples(fasta_path):
    """
    Reads a .fasta file and returns a list of (header, sequence) tuples.

    Args:
        fasta_path (str): Path to the .fasta file.

    Returns:
        list: A list of tuples, where each tuple contains the sequence header
              (without the '>' symbol) and the sequence string.
    """
    sequences = []
    header = None
    seq = ""

    # Open and read the fasta file line by line
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # If a header and sequence were previously collected, store them
                if header and seq:
                    sequences.append((header, seq))
                # Start a new sequence entry with the current header
                header = line[1:]  # remove the ">" character
                seq = ""
            else:
                # Append non-header lines to the current sequence
                seq += line
        # Add the last sequence in the file after the loop finishes
        if header and seq:
            sequences.append((header, seq))

    return sequences

def build_sequence_dictionary(protein_sequences_directory_path):
    """
    Reads all .fasta files in a directory and builds a dictionary mapping
    simplified serogroup names (derived from filenames to match training features)
    to lists of protein sequences.

    Args:
        protein_sequences_directory_path (str): Path to the directory containing
                                                the protein sequence .fasta files.

    Returns:
        dict: A dictionary where keys are simplified serogroup names (matching
              training features) and values are lists of (header, sequence) tuples
              parsed from the .fasta files.
    """
    protein_sequences_by_serogroup = {}

    # Iterate through each file in the specified directory
    for filename in os.listdir(protein_sequences_directory_path):
        # Process files with common FASTA extensions
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            fasta_path = os.path.join(protein_sequences_directory_path, filename)

            # Extract the simplified serogroup name from the filename
            # Assuming format like 'cl_1_Ballum_GCF_009884235.1.fasta'
            # We want to extract 'Ballum'
            parts = filename.split("_")
            if len(parts) > 2:
                simplified_serogroup_name = parts[2] # Get the part after the second underscore
                # Remove the extension and any trailing GCF/version info if present
                simplified_serogroup_name = simplified_serogroup_name.split(".")[0]
                # Handle cases like 'cl_2_Grippotyphosa.fasta'
                if simplified_serogroup_name.endswith('fasta') or simplified_serogroup_name.endswith('fa'):
                     simplified_serogroup_name = os.path.splitext(simplified_serogroup_name)[0]

            else:
                # Fallback or handle unexpected filename formats
                print(f"Warning: Unexpected filename format for extracting serogroup: {filename}. Using full name without extension as key.")
                simplified_serogroup_name = os.path.splitext(filename)[0]


            # Parse the fasta file into a list of (header, sequence) tuples
            sequence_list = parse_fasta_to_tuples(fasta_path)

            # Add the parsed sequences to the dictionary under the simplified serogroup name key
            protein_sequences_by_serogroup[simplified_serogroup_name] = sequence_list

            print(f"Processed {filename} and mapped to serogroup: {simplified_serogroup_name} ({len(sequence_list)} sequences)")

    return protein_sequences_by_serogroup

"""## Data Loading and Preprocessing Functions

This function preprocesses new data for prediction using a trained scaler.
"""

# Removed the scaler argument from the function definition
def preprocess_for_prediction(table):
    """
    Preprocess data for machine learning prediction.

    Args:
        table (pd.DataFrame): The input DataFrame for prediction.

    Returns:
        tuple: (processed_table, original_indices) or (None, None) if there is an error.
               processed_table (pd.DataFrame): Processed feature DataFrame.
               original_indices (list): List of original row indices from the input DataFrame.
    """
    print("Starting preprocessing for prediction...")
    # Drop the specified columns that are not features, ignoring errors if they don't exist
    columns_to_exclude = ['Strain', 'Serogroup', 'Species','max']
    processed_table = table.drop(columns=columns_to_exclude, errors='ignore').copy()
    print(f"  - Columns '{columns_to_exclude}' removed. Current shape: {processed_table.shape}")

    # Ensure all columns are numeric, coercing non-numeric values to NaN
    print("  - Converting non-numeric columns to numeric...")
    non_numeric_columns = processed_table.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric_columns:
        print(f"    - Non-numeric columns detected: {non_numeric_columns}")
        for col in non_numeric_columns:
             processed_table[col] = pd.to_numeric(processed_table[col], errors='coerce')
    print("  - Column conversion completed.")

    # Drop rows with any remaining missing values (NaNs)
    rows_before_dropna = processed_table.shape[0]
    processed_table = processed_table.dropna()
    rows_after_dropna = processed_table.shape[0]
    if rows_after_dropna < rows_before_dropna:
        print(f"  - Removed {rows_before_dropna - rows_after_dropna} rows with missing values.")
    print(f"  - Shape after removing missing values: {processed_table.shape}")

    # Drop the 'class' column if it exists, as it's not a feature for prediction
    if 'class' in processed_table.columns:
        print("  - 'class' column found. Removing...")
        processed_table = processed_table.drop(columns=['class'])
        print(f"  - 'class' column removed. Current shape: {processed_table.shape}")
    else:
        print("  - 'class' column not found. Proceeding.")

    # Debugging: Print columns before prediction to verify
    print("\n  - Columns in DataFrame before prediction:")
    print(processed_table.columns.tolist())
    print(f"  - Número de colunas antes de predição: {processed_table.shape[1]}")

    # Ensure the order of columns matches the training data features.
    # This is crucial for consistent prediction.
    # A robust approach requires saving and loading the list of training feature column names.
    # Assuming for now that the remaining columns are in the correct relative order.

    # Removed scaler application

    print("Preprocessing for prediction completed.")
    # Return the feature matrix and the list of original indices for the rows that were kept
    return processed_table, processed_table.index.tolist()

"""## Loading Saved Model and Scaler and Protein Sequences

This section loads the previously trained SVM model, the corresponding scaler, and the protein sequences for BLAST.
"""

# Check if the training data file exists
if not os.path.exists(training_file_path):
    print(f"Error: Training data file not found at {training_file_path}")
    # Set variables to None or empty to indicate failure
    training_feature_columns = None
    loaded_model = None
    # loaded_scaler = None # Remove the scaler variable
    protein_sequences_by_serogroup = {}
else:
    # Load the data from the file path into a DataFrame to get training feature columns
    print(f"Loading training data from: {os.path.basename(training_file_path)} to identify features.")
    try:
        df_from_file = pd.read_csv(training_file_path)

        # Process the training data to identify the feature columns used during training.
        # This assumes the 'verificar_e_transformar_dados' function processes the data
        # in a way that the resulting DataFrame's columns (excluding the label) are the features.
        # NOTE: The 'verificar_e_transformar_dados' function is not present in the provided cells.
        # This part of the code might need adjustment based on how you identify training features.
        # As a temporary placeholder, let's assume we can determine features after dropping
        # Strain, Serogroup, Species, and class.

        temp_df_processed = df_from_file.copy()
        # Drop columns that are not features based on common non-feature columns and potentially 'class'
        cols_to_drop_for_features = ['Strain', 'Serogroup', 'Species', 'class'] # Include 'class' if it's in the training data
        # Filter columns to ensure they exist in the DataFrame before dropping
        existing_cols_to_drop = [col for col in cols_to_drop_for_features if col in temp_df_processed.columns]
        training_feature_columns = [col for col in temp_df_processed.columns if col not in existing_cols_to_drop]
        # Remove 'max' from the training feature columns if it exists
        if 'max' in training_feature_columns:
             training_feature_columns.remove('max')


        print(f"\nIdentified {len(training_feature_columns)} training feature columns based on exclusion.")
        # print("Training feature columns:", training_feature_columns) # Uncomment for debugging

        # Define the base name of the training file without extension
        # training_base_name = os.path.splitext(os.path.basename(training_file_path))[0]
        # NOTE: The model and scaler filenames might not directly correspond to the training data filename.
        # It's safer to define their names explicitly or based on how they were saved.
        # Assuming the saved model is 'modelo_sorogrupo.pkl' and scaler is 'scaler.pkl' as per the download list.
        # Corrected filenames based on the actual files in the cloned repository
        model_filename = "averages_with_serogroups_species - MLclass_model.pkl"



        # Define the path to the directory where the model and scaler are saved.
        # NOTE: This path is now set in the initial setup cell after downloading.
        # model_directory_path = "/content/drive/MyDrive/" # Replace with your actual model directory path

        # Define the full path to the saved model file
        # Assuming the model file name is '<training_base_name>_model.pkl'
        model_load_path = os.path.join(model_directory_path, model_filename)

        # Load the saved model
        # Check if the model file exists before attempting to load
        if not os.path.exists(model_load_path):
            print(f"Error: Model file not found at {model_load_path}")
            loaded_model = None
        else:
            print(f"Loading model from: {model_load_path}")
            loaded_model = joblib.load(model_load_path)
            print("Model loaded successfully.")

         # Check if the protein sequences directory exists
        if not os.path.isdir(protein_sequences_directory_path):
             print(f"Error: Protein sequences directory not found at {protein_sequences_directory_path}")
             protein_sequences_by_serogroup = {} # Initialize as empty if directory not found
        else:
            # Load protein sequences using the function
            print(f"Loading protein sequences from: {protein_sequences_directory_path}")
            # Use the build_sequence_dictionary function to load sequences from the local directory
            protein_sequences_by_serogroup = build_sequence_dictionary(protein_sequences_directory_path)
            if protein_sequences_by_serogroup:
                print(f"Loaded protein sequences for {len(protein_sequences_by_serogroup)} serogroups.")
            else:
                print("No protein sequences were loaded.")

    except pd.errors.EmptyDataError:
        print(f"Error: Training data file is empty at {training_file_path}")
        training_feature_columns = None
        loaded_model = None
        # loaded_scaler = None # Remove loaded_scaler
        protein_sequences_by_serogroup = {}
    except FileNotFoundError:
        print(f"Error: A required file was not found.")
        training_feature_columns = None
        loaded_model = None
        # loaded_scaler = None # Remove loaded_scaler
        protein_sequences_by_serogroup = {}
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        import traceback
        traceback.print_exc()
        training_feature_columns = None
        loaded_model = None
        # loaded_scaler = None # Remove loaded_scaler
        protein_sequences_by_serogroup = {}


# Add a check to ensure necessary components are loaded before proceeding
# Removed loaded_scaler from the check
if (loaded_model is None or training_feature_columns is None or not protein_sequences_by_serogroup):
    print("\nWarning: One or more necessary components (Model, Training Feature Columns, or Protein Sequences) could not be loaded. Please ensure the initial setup cell ran successfully and the required files were downloaded.")
else:
     print("\nAll necessary components (Model, Training Feature Columns, Protein Sequences) are loaded and ready for prediction.")
## Gradio Interface
# --- Define the wrapper function for Gradio ---
def predict_from_files(files):
    results = []
    # Initial status message - will be updated by the first yield
    status_message = "Starting prediction process..."

    # Yield the initial status update immediately
    yield pd.DataFrame(), gr.update(visible=False), gr.update(value=status_message, visible=True)


    if not files:
        # Return empty DataFrame, hide output, and update status
        yield pd.DataFrame(), gr.update(visible=False), gr.update(value="⚠️ No files uploaded.", visible=True)
        return # Exit the generator

    for f in files:
        try:
            # Update status with the current file being processed
            status_message = f"Processing {os.path.basename(f.name)}..."
            # Yield the status update immediately
            yield pd.DataFrame(), gr.update(visible=False), gr.update(value=status_message, visible=True)

            prediction_df = predict_serogroup_from_genome(
                f.name,
                protein_sequences_by_serogroup,
                loaded_model,
                training_feature_columns
            )
            if prediction_df is not None:
                # Calculate confidence (highest probability)
                if 'Predicted_Class' in prediction_df.columns and hasattr(loaded_model, 'classes_'):
                    # Find the column name corresponding to the predicted class probability
                    predicted_class = prediction_df['Predicted_Class'].iloc[0]
                    prob_col_name = f'Probability_Class_{predicted_class}'
                    if prob_col_name in prediction_df.columns:
                        prediction_df['Confidence'] = prediction_df[prob_col_name]
                        # Reorder columns to place Confidence next to Predicted_Class
                        cols = prediction_df.columns.tolist()
                        try:
                            predicted_class_index = cols.index('Predicted_Class')
                            confidence_index = cols.index('Confidence')
                            # Move 'Confidence' after 'Predicted_Class'
                            cols.insert(predicted_class_index + 1, cols.pop(confidence_index))
                            prediction_df = prediction_df[cols]
                        except ValueError:
                            # If columns are not found, just append Confidence at the end
                            pass
                results.append(prediction_df)

        except Exception as e:
            results.append(pd.DataFrame({"Error": [str(e)], "File": [f.name]}))
            # Update status on error
            status_message = f"❌ Error processing {os.path.basename(f.name)}: {str(e)}"
            yield pd.DataFrame(), gr.update(visible=False), gr.update(value=status_message, visible=True)


    if results:
        combined_results = pd.concat(results, ignore_index=True)
        # Return results, make output visible, and clear status
        status_message = "✅ Prediction complete."
        yield combined_results, gr.update(visible=True), gr.update(value=status_message, visible=True)
    else:
        # Return empty DataFrame, hide output, and update status if no successful predictions were made
        status_message = "❌ No successful predictions were made."
        yield pd.DataFrame(), gr.update(visible=False), gr.update(value=status_message, visible=True)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Leptospira Classifier Prototype")
    gr.Markdown("Upload `.fna` or `.fas` files and predict the **class/serogroup**.")

    with gr.Row():
        file_input = gr.File(file_types=[".fna", ".fas"], type="filepath", file_count="multiple", label="Upload genome files")

    with gr.Row():
        predict_btn = gr.Button("Predict Class")

    # Add a Textbox for status messages - initially visible but empty
    status_output = gr.Textbox(label="Status", interactive=False, value="", visible=True)

    # Set initial value to None and visible to False for the output table
    output = gr.Dataframe(label="Prediction Results", value=None, visible=False)

    # Button actions - now update the output DataFrame, its visibility, AND the status Textbox
    # Setting 'api_name=False' prevents this function from being exposed as an API endpoint
    predict_btn.click(predict_from_files, inputs=file_input, outputs=[output, output, status_output], api_name=False)

# Launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)