import os
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class SnomedRelations:

    def __init__(
        self,
        medcat=False,
        snowstorm=False,
        aliencat=False,
        dgx=False,
        dhcap=False,
        dhcap02=True,
        snomed_rf2_full_path=None,
        medcat_path=None,
    ):

        sys.path.insert(0, "..")

        snomed_default = os.environ.get(
            "SNOMED_RF2_PATH",
            "/data/snomed/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/Full/Terminology/sct2_StatedRelationship_Full_INT_20231101.txt",
        )

        if snomed_rf2_full_path is None:
            snomed_rf2_full_path = snomed_default

        if not os.path.exists(snomed_rf2_full_path):
            raise FileNotFoundError(
                f"SNOMED RF2 path not found: {snomed_rf2_full_path}. "
                f"Set SNOMED_RF2_PATH environment variable to override the default."
            )

        self.df = pd.read_csv(snomed_rf2_full_path, sep="\t", header=0)

        self.medcat = medcat

        self.snowstorm = snowstorm

        if self.medcat:
            try:
                from medcat.cat import CAT
            except ImportError:
                self.cat = None

            else:
                medcat_default = os.environ.get(
                    "MEDCAT_MODEL_PATH",
                    "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip",
                )
                medcat_path = self._get_medcat_path(
                    medcat_path,
                    medcat_default,
                    aliencat,
                    dgx,
                    dhcap,
                    dhcap02,
                )
                try:
                    self.cat = CAT.load_model_pack(medcat_path)
                except (FileNotFoundError, OSError):
                    self.cat = None
        else:
            self.cat = None

    def _get_medcat_path(
        self,
        medcat_path,
        medcat_default,
        aliencat=False,
        dgx=False,
        dhcap=False,
        dhcap02=False,
    ):
        if aliencat:
            return self._get_aliencat_path(medcat_path, medcat_default)
        if dgx:
            return self._get_dgx_path(medcat_path)
        if dhcap:
            return self._get_dhcap_path(medcat_path)
        if dhcap02:
            return self._get_dhcap02_path(medcat_path)
        if medcat_path is None:
            return medcat_default
        return medcat_path

    def _get_aliencat_path(self, medcat_path, medcat_default):
        medcat_path_env = os.environ.get("MEDCAT_ALIENCAT_PATH")
        if medcat_path is None:
            medcat_path = medcat_default
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dgx_path(self, medcat_path):
        medcat_path_env = os.environ.get("MEDCAT_DGX_PATH")
        if medcat_path is None:
            medcat_path = (
                "/data/medcat_models/20230328_trained_model_hfe_redone/"
                "medcat_model_pack_316666b47dfaac07"
            )
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dhcap_path(self, medcat_path):
        medcat_path_env = os.environ.get("MEDCAT_DHCAP_PATH")
        if medcat_path is None:
            medcat_path = "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip"
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dhcap02_path(self, medcat_path):
        medcat_path_env = os.environ.get("MEDCAT_DHCAP02_PATH")
        if medcat_path is None:
            medcat_path = "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip"
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def get_children(self, cui):
        try:
            cui = int(cui)
            return self.df[self.df["destinationId"] == cui]["sourceId"].to_list()
        except ValueError:
            return []

    def get_parents(self, cui):
        try:
            cui = int(cui)
            return self.df[self.df["sourceId"] == cui]["destinationId"].to_list()
        except ValueError:
            return []

    def expand_codes(self, filter_root_cui, debug=False):

        if self.snowstorm:
            return self.expand_codes_snowstorm(filter_root_cui, debug=debug)

        return self.expand_codes_local(filter_root_cui, debug=debug)

    # def expand_codes_local(self, filter_root_cui, debug = False):

    #     if debug:

    #         print("Entering expand_codes_local function")
    #         print(f"filter_root_cui: {filter_root_cui}")

    #     retrieved_codes_temp = []
    #     retrieved_names_temp = []

    #     cr = self.expand_codes_parents_local(filter_root_cui, debug=debug)

    #     if debug:
    #         print(f"cr: {len(cr)}")

    #     ar = self.expand_codes_children_local(filter_root_cui, debug=debug)

    #     if debug:
    #         print(f"ar: {len(ar)}")

    #     retrieved_codes_temp.extend(cr)
    #     retrieved_codes_temp.extend(ar)
    #     retrieved_names_temp.extend(self.get_pretty_name_list(cr))
    #     retrieved_names_temp.extend(self.get_pretty_name_list(ar))
    #     retrieved_codes_temp = list(set(retrieved_codes_temp))
    #     retrieved_names_temp = list(set(retrieved_names_temp))

    #     if debug:
    #         print(
    #             f"{len(retrieved_codes_temp)} retrieved_codes:"
    #             f" {retrieved_codes_temp}"
    #         )

    #     return retrieved_codes_temp, retrieved_names_temp

    def expand_codes_local(self, filter_root_cui, debug=False):

        if debug:

            pass

        retrieved_codes_temp = []
        retrieved_names_temp = []

        cr = self.expand_codes_parents_local(filter_root_cui, debug=debug)

        if debug:
            pass

        ar = self.expand_codes_children_local(filter_root_cui, debug=debug)

        if debug:
            pass

        retrieved_codes_temp.extend(cr[0])
        retrieved_codes_temp.extend(ar[0])
        retrieved_names_temp.extend(cr[1])
        retrieved_names_temp.extend(ar[1])
        retrieved_codes_temp = list(set(retrieved_codes_temp))
        retrieved_names_temp = list(set(retrieved_names_temp))

        if debug:
            pass

        return retrieved_codes_temp, retrieved_names_temp

    def has_medcat(self):
        return (
            getattr(self, "medcat", False)
            and hasattr(self, "cat")
            and self.cat is not None
        )

    def get_pretty_name(self, cui):
        if not self.has_medcat():
            import warnings

            warnings.warn(
                f"MedCAT not available, returning None for cui: {cui}", stacklevel=2
            )
            return None
        return self.cat.cdb.cui2preferred_name.get(str(cui))

    def get_pretty_name_list(self, cui_list):
        if not self.has_medcat():
            return [None] * len(cui_list)

        pretty_name_list = []

        for i in range(0, len(cui_list)):
            name = self.get_pretty_name(cui_list[i])
            if name is not None:
                pretty_name_list.append(name)
        return pretty_name_list

    def expand_codes_children_local(self, filter_root_cui, debug=False):

        if debug:

            pass

        retrieved_names_temp = []

        children_codes = self.get_children(filter_root_cui)

        if self.medcat:
            retrieved_names_temp = self.get_pretty_name_list(children_codes)

        return children_codes, retrieved_names_temp

    def expand_codes_parents_local(self, filter_root_cui, debug=False):

        if debug:

            pass

        retrieved_names_temp = []

        parent_codes = self.get_parents(filter_root_cui)

        if self.medcat:
            retrieved_names_temp = self.get_pretty_name_list(parent_codes)

        return parent_codes, retrieved_names_temp

    def expand_codes_snowstorm(self, filter_root_cui, debug=False):
        # debug = True
        if debug:

            pass

        retrieved_codes_temp = []
        retrieved_names_temp = []

        c = self.get_snowstorm_response_children(str(filter_root_cui))
        if debug:
            pass

        a = self.get_snowstorm_response_ancestors(str(filter_root_cui))
        if debug:
            pass

        cr = self.parse_snowstorm_response_to_cui_name(c)
        if debug:
            pass

        ar = self.parse_snowstorm_response_to_cui_name(a)
        if debug:
            pass

        retrieved_codes_temp.extend(cr[0])
        retrieved_codes_temp.extend(ar[0])
        retrieved_names_temp.extend(cr[1])
        retrieved_names_temp.extend(ar[1])
        retrieved_codes_temp = list(set(retrieved_codes_temp))
        retrieved_names_temp = list(set(retrieved_names_temp))

        if debug:
            pass

        return retrieved_codes_temp, retrieved_names_temp

    def recursive_code_expansion(self, filter_root_cui, n_recursion=3, debug=False):
        retrieved_codes = [filter_root_cui]
        retrieved_names = []

        for _i in tqdm(range(n_recursion)):

            if debug:
                pass

            retrieved_codes = list(set(retrieved_codes))

            for j in range(0, len(retrieved_codes)):
                current_filter_root_cui = retrieved_codes[j]

                raw = self.expand_codes_local(
                    filter_root_cui=current_filter_root_cui, debug=debug
                )

                codes = raw[0]
                names = raw[1]

                retrieved_codes.extend(codes)
                retrieved_names.extend(names)

                retrieved_codes = list(set(retrieved_codes))
                retrieved_names = list(set(retrieved_names))

        if debug:

            pass

        return retrieved_codes, retrieved_names

    def get_medcat_cdb_most_similar(
        self, cui, context_type="xxxlong", type_id_filter=None, topn=10
    ):
        if not self.has_medcat():
            import warnings

            warnings.warn("MedCAT not available, returning empty lists", stacklevel=2)
            return [], []
        if type_id_filter is None:
            type_id_filter = []
        try:
            res = self.cat.cdb.most_similar(
                cui, context_type=context_type, type_id_filter=type_id_filter, topn=topn
            )
        except Exception:
            return [], []

        names = []
        codes = []

        key_list = list(res.keys())

        for elem in key_list:

            codes.append(elem)

        names = self.get_pretty_name_list(codes)

        return codes, names

    def build_lists_medcat_snomedtree(self, input_list, medcat=False, snomed=True):
        retrieved_codes = []
        retrieved_names = []

        if snomed:
            for item in input_list:
                codes, names = self.recursive_code_expansion(item)
                retrieved_codes.extend(codes)
                retrieved_names.extend(names)

        if medcat:
            for item in input_list:
                codes, names = self.get_medcat_cdb_most_similar(item)
                retrieved_codes.extend(codes)
                retrieved_names.extend(names)

        retrieved_codes = [int(code) for code in retrieved_codes]

        return retrieved_codes, retrieved_names

        # type-id t-16 etc
        # {T-38}    198890
        # {T-40}    173894
        # {T-11}     77284
        # {T-39}     64291
        # {T-18}     44201
        # {T-35}     34778
        # {T-6}      32944
        # {T-55}     27626
        # {T-33}     15117
        # {T-42}     14591

    def get_medcat_similar_score(self, input_cui, target_cui_list, debug=False):
        if not self.has_medcat():
            import warnings

            warnings.warn(
                "MedCAT not available, returning list of None values", stacklevel=2
            )
            return [None] * len(target_cui_list)
        if debug:
            pass

        target_cui_list = list(map(str, target_cui_list))

        input_cui = str(input_cui)

        try:
            res = self.cat.cdb.most_similar(
                input_cui, context_type="xxxlong", type_id_filter=[], topn=999999
            )
        except Exception:
            return []

        results_list = []

        if debug:
            pass

        for i in range(len(target_cui_list)):
            target_cui = target_cui_list[i]

            sim_res = res.get(target_cui)

            if sim_res is not None:

                sim_score = sim_res.get("sim")

            else:

                sim_score = None

            if sim_score is not None:

                results_list.append(sim_score)
            else:
                results_list.append(np.nan)

        if debug:
            pass

        return results_list

    def append_concept_sim_to_df(self, df, target_concept_sim_list, target_cui_list):
        if not self.has_medcat():
            import warnings

            warnings.warn(
                "MedCAT not available, skipping append_concept_sim_to_df", stacklevel=2
            )
            return df

        for elem in target_concept_sim_list:

            df[f"{elem}_concept_sim"] = self.get_medcat_similar_score(
                elem, target_cui_list, debug=True
            )

        return df

    def retrieve_search_synonyms(
        self,
        filter_root_cui,
        n_recursion=10,
        context_type="xxxlong",
        type_id_filter=None,
        topn=50,
        debug=False,
        use_snomed=True,
        use_medcat=True,
    ):
        # Initialize a list to store names
        if type_id_filter is None:
            type_id_filter = []
        all_names = []

        # Retrieve data for snomed_tree if use_snomed is True
        retrieved_codes_snomed_tree, retrieved_names_snomed_tree = (
            ([], [])
            if not use_snomed
            else self.recursive_code_expansion(
                filter_root_cui, n_recursion=n_recursion, debug=debug
            )
        )

        # Add names to the list, stripping anything in parentheses
        all_names.extend(
            [
                re.sub(r"\([^)]*\)", "", name).strip()
                for name in retrieved_names_snomed_tree
                if name is not None
            ]
        )

        # Retrieve data for medcat_cdb if use_medcat is True
        retrieved_codes_medcat_cdb, retrieved_names_medcat_cdb = (
            ([], [])
            if not use_medcat
            else self.get_medcat_cdb_most_similar(
                filter_root_cui,
                context_type=context_type,
                type_id_filter=type_id_filter,
                topn=topn,
            )
        )

        # Add names to the list, stripping anything in parentheses
        all_names.extend(
            [
                re.sub(r"\([^)]*\)", "", name).strip()
                for name in retrieved_names_medcat_cdb
                if name is not None
            ]
        )

        return (
            retrieved_codes_snomed_tree,
            retrieved_names_snomed_tree,
            retrieved_codes_medcat_cdb,
            retrieved_names_medcat_cdb,
            all_names,
        )

    def retrieve_search_synonyms_multi(
        self,
        filter_root_cui_list: List[str],
        n_recursion: int = 10,
        context_type: str = "xxxlong",
        type_id_filter: List[int] = None,
        topn: int = 50,
        debug: bool = False,
        use_snomed: bool = True,
        use_medcat: bool = True,
    ) -> Tuple[
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[str],
        List[str],
    ]:
        """
        Retrieves search synonyms for multiple filter_root_cui values.

        Args:
            filter_root_cui_list (List[str]): List of filter_root_cui values to
                retrieve search synonyms for.
            n_recursion (int, optional): Number of recursion levels for code expansion.
                Defaults to 10.
            context_type (str, optional): Type of context. Defaults to 'xxxlong'.
            type_id_filter (List[int], optional): List of type IDs for filtering.
                Defaults to [].
            topn (int, optional): Top N results to retrieve. Defaults to 50.
            debug (bool, optional): Enable debugging. Defaults to False.
            use_snomed (bool, optional): Use SNOMED for retrieval. Defaults to True.
            use_medcat (bool, optional): Use MedCAT for retrieval. Defaults to True.

        Returns:
            Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]],
                List[str], List[str]]: A tuple containing:
                - List of retrieved codes from SNOMED for each filter_root_cui.
                - List of retrieved names from SNOMED for each filter_root_cui.
                - List of retrieved codes from MedCAT for each filter_root_cui.
                - List of retrieved names from MedCAT for each filter_root_cui.
                - Flat list of all retrieved names.
                - Flat list of all retrieved codes.
        """
        # Initialize lists to store results
        if type_id_filter is None:
            type_id_filter = []
        all_retrieved_codes_snomed_tree = []
        all_retrieved_names_snomed_tree = []
        all_retrieved_codes_medcat_cdb = []
        all_retrieved_names_medcat_cdb = []
        all_names = []
        all_codes = []  # New list to store all codes

        # Iterate over each filter_root_cui in the list
        for filter_root_cui in filter_root_cui_list:
            # Retrieve data for snomed_tree if use_snomed is True
            retrieved_codes_snomed_tree, retrieved_names_snomed_tree = (
                ([], [])
                if not use_snomed
                else self.recursive_code_expansion(
                    filter_root_cui, n_recursion=n_recursion, debug=debug
                )
            )

            # Add retrieved data to the lists
            all_retrieved_codes_snomed_tree.append(retrieved_codes_snomed_tree)
            all_retrieved_names_snomed_tree.append(retrieved_names_snomed_tree)

            # Add names to the all_names list, stripping anything in parentheses
            all_names.extend(
                [
                    re.sub(r"\([^)]*\)", "", name).strip()
                    for name in retrieved_names_snomed_tree
                    if name is not None
                ]
            )

            # Add codes to the all_codes list
            all_codes.extend(retrieved_codes_snomed_tree)

            # Retrieve data for medcat_cdb if use_medcat is True
            retrieved_codes_medcat_cdb, retrieved_names_medcat_cdb = (
                ([], [])
                if not use_medcat
                else self.get_medcat_cdb_most_similar(
                    filter_root_cui,
                    context_type=context_type,
                    type_id_filter=type_id_filter,
                    topn=topn,
                )
            )

            # Add retrieved data to the lists
            all_retrieved_codes_medcat_cdb.append(retrieved_codes_medcat_cdb)
            all_retrieved_names_medcat_cdb.append(retrieved_names_medcat_cdb)

            # Add names to the all_names list, stripping anything in parentheses
            all_names.extend(
                [
                    re.sub(r"\([^)]*\)", "", name).strip()
                    for name in retrieved_names_medcat_cdb
                    if name is not None
                ]
            )

            # Add codes to the all_codes list
            all_codes.extend(retrieved_codes_medcat_cdb)

            all_codes = [str(code) for code in all_codes]

        return (
            all_retrieved_codes_snomed_tree,
            all_retrieved_names_snomed_tree,
            all_retrieved_codes_medcat_cdb,
            all_retrieved_names_medcat_cdb,
            all_names,
            all_codes,
        )

    def get_snowstorm_response_children(self, concept_id):
        url = f"https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN%2FSNOMEDCT-GB/concepts/{concept_id}/children?form=inferred&includeDescendantCount=false"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            ),
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            return None

    def get_snowstorm_response_ancestors(self, concept_id):
        url = f"https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN%2FSNOMEDCT-GB/concepts/{concept_id}/ancestors?excludeDirectChild=false"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            ),
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            return None

    def parse_snowstorm_response_to_cui_name(self, response_text):
        import json

        if response_text is None:
            return [], []

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return [], []

        codes = []
        names = []

        items = data.get("children", [])
        for item in items:
            concept_id = item.get("conceptId")
            if concept_id:
                codes.append(concept_id)
                preferred_term = item.get("preferredTerm", "")
                names.append(preferred_term)

        return codes, names
