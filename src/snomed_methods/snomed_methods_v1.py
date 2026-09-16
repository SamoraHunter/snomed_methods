import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class SnomedRelations:

    def __init__(
        self,
        medcat: bool = False,
        snowstorm: bool = False,
        aliencat: bool = False,
        dgx: bool = False,
        dhcap: bool = False,
        dhcap02: bool = True,
        snomed_rf2_full_path: Optional[str] = None,
        medcat_path: Optional[str] = None,
    ) -> None:

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

        self.df: pd.DataFrame = pd.read_csv(snomed_rf2_full_path, sep="\t", header=0)

        self.medcat: bool = medcat

        self.snowstorm: bool = snowstorm

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
        medcat_path: Optional[str],
        medcat_default: str,
        aliencat: bool = False,
        dgx: bool = False,
        dhcap: bool = False,
        dhcap02: bool = False,
    ) -> str:
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

    def _get_aliencat_path(
        self, medcat_path: Optional[str], medcat_default: str
    ) -> str:
        medcat_path_env = os.environ.get("MEDCAT_ALIENCAT_PATH")
        if medcat_path is None:
            medcat_path = medcat_default
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dgx_path(self, medcat_path: Optional[str]) -> str:
        medcat_path_env = os.environ.get("MEDCAT_DGX_PATH")
        if medcat_path is None:
            medcat_path = (
                "/data/medcat_models/20230328_trained_model_hfe_redone/"
                "medcat_model_pack_316666b47dfaac07"
            )
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dhcap_path(self, medcat_path: Optional[str]) -> str:
        medcat_path_env = os.environ.get("MEDCAT_DHCAP_PATH")
        if medcat_path is None:
            medcat_path = "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip"
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dhcap02_path(self, medcat_path: Optional[str]) -> str:
        medcat_path_env = os.environ.get("MEDCAT_DHCAP02_PATH")
        if medcat_path is None:
            medcat_path = "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip"
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def get_children(self, cui: Union[int, str]) -> List[int]:
        try:
            cui = int(cui)
            return self.df[self.df["destinationId"] == cui]["sourceId"].to_list()
        except ValueError:
            return []

    def get_parents(self, cui: Union[int, str]) -> List[int]:
        try:
            cui_int = int(cui)
            result = self.df[self.df["sourceId"] == cui_int]["destinationId"]
            return result.tolist()
        except ValueError:
            return []

    def expand_codes(
        self, filter_root_cui: Union[int, str], debug: bool = False
    ) -> Tuple[List[int], List[str]]:

        if self.snowstorm:
            return self.expand_codes_snowstorm(filter_root_cui, debug=debug)

        return self.expand_codes_local(filter_root_cui, debug=debug)

    def expand_codes_local(
        self, filter_root_cui: Union[int, str], debug: bool = False
    ) -> Tuple[List[int], List[str]]:

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

    def has_medcat(self) -> bool:
        return (
            getattr(self, "medcat", False)
            and hasattr(self, "cat")
            and self.cat is not None
        )

    def get_pretty_name(self, cui: Union[int, str]) -> Optional[str]:
        if not self.has_medcat():
            import warnings

            warnings.warn(
                f"MedCAT not available, returning None for cui: {cui}", stacklevel=2
            )
            return None
        return self.cat.cdb.cui2preferred_name.get(str(cui))

    def get_pretty_name_list(
        self, cui_list: List[Union[int, str]]
    ) -> List[Optional[str]]:
        if not self.has_medcat():
            return [None] * len(cui_list)

        pretty_name_list: List[Optional[str]] = []

        for i in range(0, len(cui_list)):
            name = self.get_pretty_name(cui_list[i])
            if name is not None:
                pretty_name_list.append(name)
        return pretty_name_list

    def expand_codes_children_local(
        self, filter_root_cui: Union[int, str], debug: bool = False
    ) -> Tuple[List[int], List[str]]:

        if debug:

            pass

        retrieved_names_temp = []

        children_codes = self.get_children(filter_root_cui)

        if self.medcat:
            retrieved_names_temp = self.get_pretty_name_list(children_codes)

        return children_codes, retrieved_names_temp

    def expand_codes_parents_local(
        self, filter_root_cui: Union[int, str], debug: bool = False
    ) -> Tuple[List[int], List[str]]:

        if debug:

            pass

        retrieved_names_temp = []

        parent_codes = self.get_parents(filter_root_cui)

        if self.medcat:
            retrieved_names_temp = self.get_pretty_name_list(parent_codes)

        return parent_codes, retrieved_names_temp

    def expand_codes_snowstorm(
        self, filter_root_cui: Union[int, str], debug: bool = False
    ) -> Tuple[List[int], List[str]]:
        if debug:
            pass

        retrieved_codes_temp: List[int] = []
        retrieved_names_temp: List[str] = []

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

    def recursive_code_expansion(
        self,
        filter_root_cui: Union[int, str],
        n_recursion: int = 3,
        debug: bool = False,
    ) -> Tuple[List[int], List[str]]:
        retrieved_codes: List[Union[int, str]] = [filter_root_cui]
        retrieved_names: List[str] = []

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
        self,
        cui: Union[int, str],
        context_type: str = "xxxlong",
        type_id_filter: Optional[List[int]] = None,
        topn: int = 10,
    ) -> Tuple[List[str], List[Optional[str]]]:
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

    def build_lists_medcat_snomedtree(
        self,
        input_list: List[Union[int, str]],
        medcat: bool = False,
        snomed: bool = True,
    ) -> Tuple[List[int], List[Optional[str]]]:
        retrieved_codes: List[int] = []
        retrieved_names: List[str] = []

        if snomed:
            for item in input_list:
                codes, names = self.recursive_code_expansion(item)
                retrieved_codes.extend(codes)
                retrieved_names.extend(names)

        if medcat:
            for item in input_list:
                codes, names = self.get_medcat_cdb_most_similar(item)
                retrieved_codes.extend([int(code) for code in codes])
                retrieved_names.extend(names)

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

    def get_medcat_similar_score(
        self,
        input_cui: Union[int, str],
        target_cui_list: List[Union[int, str]],
        debug: bool = False,
    ) -> List[Optional[float]]:
        if not self.has_medcat():
            import warnings

            warnings.warn(
                "MedCAT not available, returning list of None values", stacklevel=2
            )
            return [None] * len(target_cui_list)
        if debug:
            pass

        target_cui_list_str = list(map(str, target_cui_list))

        input_cui_str = str(input_cui)

        try:
            res = self.cat.cdb.most_similar(
                input_cui_str, context_type="xxxlong", type_id_filter=[], topn=999999
            )
        except Exception:
            return []

        results_list: List[Optional[float]] = []

        if debug:
            pass

        for i in range(len(target_cui_list)):
            target_cui = target_cui_list_str[i]

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

    def append_concept_sim_to_df(
        self,
        df: pd.DataFrame,
        target_concept_sim_list: List[Union[int, str]],
        target_cui_list: List[Union[int, str]],
    ) -> pd.DataFrame:
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
        filter_root_cui: Union[int, str],
        n_recursion: int = 10,
        context_type: str = "xxxlong",
        type_id_filter: Optional[List[int]] = None,
        topn: int = 50,
        debug: bool = False,
        use_snomed: bool = True,
        use_medcat: bool = True,
    ) -> Tuple[
        List[str],
        List[str],
        List[str],
        List[Optional[str]],
        List[str],
    ]:
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

    def get_snowstorm_response_children(
        self, concept_id: Union[int, str]
    ) -> Optional[str]:
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

    def get_snowstorm_response_ancestors(
        self, concept_id: Union[int, str]
    ) -> Optional[str]:
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

    def parse_snowstorm_response_to_cui_name(
        self, response_text: Optional[str]
    ) -> Tuple[List[str], List[str]]:
        import json

        if response_text is None:
            return [], []

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return [], []

        codes: List[str] = []
        names: List[str] = []

        items = data.get("children", [])
        for item in items:
            concept_id = item.get("conceptId")
            if concept_id:
                codes.append(concept_id)
                preferred_term = item.get("preferredTerm", "")
                names.append(preferred_term)

        return codes, names

    def get_subsumed_concepts(
        self,
        cui: Union[int, str],
        semantic_tags: Optional[List[str]] = None,
        max_depth: int = 10,
        active_only: bool = True,
        include_ancestors: bool = False,
        include_descendants: bool = True,
    ) -> Tuple[List[int], List[str]]:
        """Calculate transitive closure over is_a relationships to retrieve
        entire active DAG subgraph for a concept.

        Args:
            cui: Root concept ID to start traversal from.
            semantic_tags: List of semantic tags to filter by (e.g., ['disorder',
                          'finding']). If None, no filtering applied.
            max_depth: Maximum recursion depth.
            active_only: Only include active concepts and relationships.
            include_ancestors: Whether to traverse upward (parents).
            include_descendants: Whether to traverse downward (children).

        Returns:
            Tuple of (concept_ids, concept_names) lists.
        """
        try:
            cui = int(cui)
        except (ValueError, TypeError):
            return [], []

        if max_depth < 0:
            return [], []

        visited_concepts, all_concepts, depth_map = self._subsumed_traverse(
            cui,
            max_depth=max_depth,
            active_only=active_only,
            include_ancestors=include_ancestors,
            include_descendants=include_descendants,
        )

        concept_ids = list(set(all_concepts))

        if semantic_tags and self.has_medcat():
            concept_ids = self._filter_by_semantic_tags(concept_ids, semantic_tags)

        concept_names = self.get_pretty_name_list(concept_ids)

        return concept_ids, concept_names

    def _subsumed_traverse(
        self,
        cui: int,
        max_depth: int,
        active_only: bool,
        include_ancestors: bool,
        include_descendants: bool,
    ) -> Tuple[Set[int], List[int], Dict[int, int]]:
        """Traverse the DAG and return visited concepts."""
        visited_concepts = set()
        all_concepts = [cui]
        depth_map = {cui: 0}

        def traverse(concept_id: int, depth: int):
            if depth > max_depth or concept_id in visited_concepts:
                return

            visited_concepts.add(concept_id)
            all_concepts.append(concept_id)
            depth_map[concept_id] = depth

            df = self._get_active_df() if active_only else self.df
            type_id = 116680003

            children_df = (
                df[(df["destinationId"] == concept_id) & (df["typeId"] == type_id)]
                if include_descendants
                else pd.DataFrame()
            )

            parents_df = (
                df[(df["sourceId"] == concept_id) & (df["typeId"] == type_id)]
                if include_ancestors
                else pd.DataFrame()
            )

            next_level_df = pd.concat([children_df, parents_df])

            if not next_level_df.empty:
                for _, row in next_level_df.iterrows():
                    # For children: destinationId is the parent, sourceId is the child
                    # For parents: sourceId is the child, destinationId is the parent
                    if "sourceId" in row and "destinationId" in row:
                        if include_descendants and row["destinationId"] == concept_id:
                            child_id = int(row["sourceId"])
                        elif include_ancestors and row["sourceId"] == concept_id:
                            child_id = int(row["destinationId"])
                        else:
                            child_id = (
                                int(row["sourceId"])
                                if include_descendants
                                else int(row["destinationId"])
                            )
                    else:
                        child_id = int(row.get("sourceId") or row.get("destinationId"))
                    traverse(child_id, depth + 1)

        traverse(cui, 0)
        return visited_concepts, all_concepts, depth_map

    def _get_active_df(self) -> pd.DataFrame:
        """Get active relationships dataframe."""
        if "active" in self.df.columns:
            return self.df[(self.df["active"] == "1") | (self.df["active"] == 1)]
        return self.df

    def _filter_by_semantic_tags(
        self, concept_ids: List[int], semantic_tags: List[str]
    ) -> List[int]:
        """Filter concepts by semantic tags."""
        filtered_ids = []
        for cid in concept_ids:
            name = self.get_pretty_name(cid)
            if name is not None:
                name_lower = name.lower()
                if any(tag.lower() in name_lower for tag in semantic_tags):
                    filtered_ids.append(cid)
        return filtered_ids
