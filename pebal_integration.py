import torch
import os
import logging
from typing import Dict, Any, Tuple, List, Set

# Assume HopfieldPEBALModel is defined elsewhere and inherits from torch.nn.Module
# class HopfieldPEBALModel(torch.nn.Module): ...

logger = logging.getLogger("Hopfield-PEBAL")
# Configure basic logging if running standalone for testing
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def integrate_pebal_weights(
    hopfield_model: torch.nn.Module,
    pebal_checkpoint_path: str
) -> torch.nn.Module:
    """
    Transfers weights from a pre-trained PEBAL model checkpoint to a Hopfield-PEBAL model.

    This function loads a PEBAL state dictionary, attempts to map its keys to the
    corresponding keys in the Hopfield-PEBAL model using a predefined mapping,
    and copies the weights for matching keys and shapes. It handles common
    checkpoint structures and logs detailed information about the transfer process.

    Args:
        hopfield_model: The HopfieldPEBALModel instance (or any torch.nn.Module)
                        to load weights into.
        pebal_checkpoint_path: Path to the PEBAL model checkpoint (.pth or .pt file).

    Returns:
        The updated HopfieldPEBALModel instance with transferred weights.

    Raises:
        FileNotFoundError: If the pebal_checkpoint_path does not exist.
        KeyError: If the checkpoint format is unexpected and the state dict
                  cannot be extracted.
        TypeError: If the loaded checkpoint or extracted state_dict is not a dictionary.
        RuntimeError: If torch.load fails for other reasons (e.g., corrupted file).
    """
    logger.info(f"Attempting to load PEBAL weights from: {pebal_checkpoint_path}")
    if not os.path.exists(pebal_checkpoint_path):
        logger.error(f"PEBAL checkpoint file not found at {pebal_checkpoint_path}")
        raise FileNotFoundError(f"PEBAL checkpoint not found at {pebal_checkpoint_path}")

    # --- 1. Load PEBAL checkpoint ---
    try:
        # Load onto CPU first to avoid GPU memory issues if not needed immediately
        pebal_checkpoint = torch.load(pebal_checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.error(f"Error loading PEBAL checkpoint file: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load checkpoint '{pebal_checkpoint_path}'") from e

    # --- 2. Extract state dict from checkpoint ---
    pebal_state_dict: Dict[str, Any]
    if isinstance(pebal_checkpoint, dict):
        if 'model' in pebal_checkpoint:
            pebal_state_dict = pebal_checkpoint['model']
            logger.info("Extracted state dict from 'model' key in PEBAL checkpoint.")
        elif 'state_dict' in pebal_checkpoint:
            pebal_state_dict = pebal_checkpoint['state_dict']
            logger.info("Extracted state dict from 'state_dict' key in PEBAL checkpoint.")
        else:
            # Assume the dictionary itself is the state dict if common keys aren't found
            # Add a check to ensure values are tensors, a common characteristic of state_dicts
            is_likely_state_dict = all(isinstance(v, torch.Tensor) for v in pebal_checkpoint.values())
            if is_likely_state_dict and pebal_checkpoint:
                 logger.warning("PEBAL checkpoint dictionary structure not recognized ('model' or 'state_dict' keys not found). "
                                "Assuming the dictionary itself is the state_dict.")
                 pebal_state_dict = pebal_checkpoint
            else:
                 logger.error("Could not find state dict in PEBAL checkpoint. Expected keys like 'model' or 'state_dict', or a direct state_dict.")
                 raise KeyError("Could not find state dict in PEBAL checkpoint. Check the checkpoint structure.")
    elif isinstance(pebal_checkpoint, dict): # Redundant check kept for clarity, primary check is above
         # This case is less common for checkpoints but possible
         logger.warning("PEBAL checkpoint is not a dictionary. Assuming it's the state_dict directly.")
         pebal_state_dict = pebal_checkpoint # type: ignore # Ignore type error as we assume structure
    else:
        logger.error(f"Loaded PEBAL checkpoint is not a dictionary (type: {type(pebal_checkpoint)}). Cannot extract state dict.")
        raise TypeError(f"Expected checkpoint to be a dictionary, but got {type(pebal_checkpoint)}")

    if not isinstance(pebal_state_dict, dict):
         logger.error(f"Extracted PEBAL state_dict is not a dictionary (type: {type(pebal_state_dict)}).")
         raise TypeError(f"Expected extracted state_dict to be a dictionary, but got {type(pebal_state_dict)}")


    # --- 3. Get current Hopfield model state dict ---
    hopfield_state_dict = hopfield_model.state_dict()
    hopfield_keys: Set[str] = set(hopfield_state_dict.keys())

    # --- 4. Define Key Mapping ---
    # This mapping is CRUCIAL and depends entirely on the specific architectures
    # of your PEBAL and Hopfield-PEBAL models. Adjust it accordingly.
    # Format: {pebal_prefix_or_full_key: hopfield_prefix_or_full_key}
    key_mapping: Dict[str, str] = {
        # --- Backbone Mappings ---
        # Example: If PEBAL uses 'branch1.mod1.conv.weight' and Hopfield uses 'backbone.block1.conv.weight'
        # You might map the prefixes: 'branch1.mod' -> 'backbone.block'
        'branch1.mod1': 'backbone.mod1',
        'branch1.mod2': 'backbone.mod2',
        'branch1.mod3': 'backbone.mod3',
        'branch1.mod4': 'backbone.mod4',
        'branch1.mod5': 'backbone.mod5',
        'branch1.mod6': 'backbone.mod6',
        'branch1.mod7': 'backbone.mod7',
        'branch1.pool2': 'backbone.pool2',
        'branch1.pool3': 'backbone.pool3',
        # --- Segmentation Head Mappings ---
        'branch1.aspp': 'segmentation_head.aspp',
        'branch1.bot_fine': 'segmentation_head.bot_fine',
        'branch1.bot_aspp': 'segmentation_head.bot_aspp',
        'branch1.final': 'segmentation_head.final',
        # --- Add more specific mappings if needed ---
        # 'pebal_specific_layer_name': 'hopfield_equivalent_layer_name',
    }
    logger.info(f"Using {len(key_mapping)} mapping rules for weight transfer.")

    # --- 5. Transfer Weights ---
    transferred_keys_count: int = 0
    mismatched_shape_keys: List[Tuple[str, str]] = []
    unmapped_pebal_keys: List[str] = []
    skipped_due_to_mapping_issue: List[str] = [] # Keys matched by prefix but final key invalid

    for pebal_key, pebal_param in pebal_state_dict.items():
        hopfield_key: str | None = None

        # Check if the exact PEBAL key is mapped directly
        if pebal_key in key_mapping:
            potential_hopfield_key = key_mapping[pebal_key]
            if potential_hopfield_key in hopfield_keys:
                 hopfield_key = potential_hopfield_key
                 logger.debug(f"Mapped PEBAL key '{pebal_key}' directly to '{hopfield_key}'.")

        # If not mapped directly, check for prefix mapping
        if hopfield_key is None:
            for pebal_prefix, hopfield_prefix in key_mapping.items():
                if pebal_key.startswith(pebal_prefix + '.'): # Ensure it's a real prefix boundary
                    potential_hopfield_key = pebal_key.replace(pebal_prefix, hopfield_prefix, 1) # Replace only first occurrence
                    if potential_hopfield_key in hopfield_keys:
                        hopfield_key = potential_hopfield_key
                        logger.debug(f"Mapped PEBAL key '{pebal_key}' to '{hopfield_key}' using prefix '{pebal_prefix}' -> '{hopfield_prefix}'.")
                        break # Stop searching for prefixes once a match is found
                    else:
                         # Log cases where prefix matches but the resulting key doesn't exist in the target model
                         # This often indicates an issue with the mapping definition or model structure.
                         logger.debug(f"Prefix match for '{pebal_key}' resulted in non-existent target key '{potential_hopfield_key}'. Skipping this mapping.")
                         # Keep track for summary, but allow other mappings to potentially match
                         # skipped_due_to_mapping_issue.append(pebal_key) # Removed as it could be noisy if multiple prefixes match partially

        # If no mapping found yet, try a direct name match (PEBAL key == Hopfield key)
        # This acts as a fallback if the layer names are identical and not covered by specific mappings.
        if hopfield_key is None and pebal_key in hopfield_keys:
             # Avoid applying this fallback if the key *started* with a defined pebal_prefix,
             # as it likely means the mapping was intended but incorrect.
             was_potentially_mappable = any(pebal_key.startswith(p + '.') for p in key_mapping)
             if not was_potentially_mappable:
                hopfield_key = pebal_key
                logger.debug(f"Found direct match for key (no specific mapping defined): {pebal_key}")
             else:
                 logger.debug(f"Direct match for {pebal_key} skipped because it started with a defined PEBAL prefix but mapping failed.")
                 skipped_due_to_mapping_issue.append(pebal_key) # Log these specifically

        # --- Perform transfer if a corresponding key was found ---
        if hopfield_key:
            # Final check ensuring the identified key is valid
            if hopfield_key in hopfield_state_dict:
                target_param = hopfield_state_dict[hopfield_key]
                if target_param.shape == pebal_param.shape:
                    # Use clone().detach() to ensure no gradient history is copied
                    # and the tensor is independent.
                    hopfield_state_dict[hopfield_key] = pebal_param.clone().detach()
                    transferred_keys_count += 1
                    logger.debug(f"Successfully transferred: {pebal_key} -> {hopfield_key}")
                else:
                    logger.warning(f"Shape mismatch for key {hopfield_key} "
                                  f"(mapped from PEBAL key {pebal_key}): "
                                  f"PEBAL shape {pebal_param.shape}, "
                                  f"Hopfield shape {target_param.shape}. Skipping transfer.")
                    mismatched_shape_keys.append((pebal_key, hopfield_key))
            else:
                 # Should not happen if logic above is correct, but guards against unforeseen issues.
                 logger.error(f"Internal logic error: Identified hopfield_key '{hopfield_key}' (from pebal_key '{pebal_key}') not found in hopfield_state_dict during final check.")
                 skipped_due_to_mapping_issue.append(pebal_key)

        else:
            # Key from PEBAL checkpoint was not mapped or found directly in Hopfield model
            if pebal_key not in skipped_due_to_mapping_issue: # Avoid double counting
                 unmapped_pebal_keys.append(pebal_key)
                 logger.debug(f"PEBAL key '{pebal_key}' has no defined mapping and no direct match in Hopfield model.")


    # --- 6. Log Summary ---
    logger.info("--- Weight Transfer Summary ---")
    logger.info(f"Total keys in PEBAL state_dict: {len(pebal_state_dict)}")
    logger.info(f"Successfully transferred keys: {transferred_keys_count}")
    logger.info(f"Keys with shape mismatches (skipped): {len(mismatched_shape_keys)}")
    if mismatched_shape_keys:
        for i, (pk, hk) in enumerate(mismatched_shape_keys[:5]): # Log first 5 mismatches
            logger.info(f"  - Mismatch Example {i+1}: PEBAL='{pk}' -> Hopfield='{hk}' (Shapes: {pebal_state_dict[pk].shape} vs {hopfield_state_dict[hk].shape})")
        if len(mismatched_shape_keys) > 5: logger.info("  - ... (see previous warnings for full list)")

    logger.info(f"PEBAL keys skipped due to mapping issues (e.g., prefix matched but target key invalid): {len(skipped_due_to_mapping_issue)}")
    if skipped_due_to_mapping_issue:
         for i, pk in enumerate(skipped_due_to_mapping_issue[:5]): logger.info(f"  - Mapping Issue Example {i+1}: PEBAL='{pk}'")
         if len(skipped_due_to_mapping_issue) > 5: logger.info("  - ...")

    logger.info(f"PEBAL keys unmapped and not found directly in Hopfield model: {len(unmapped_pebal_keys)}")
    if unmapped_pebal_keys:
        for i, pk in enumerate(unmapped_pebal_keys[:5]): logger.info(f"  - Unmapped Example {i+1}: PEBAL='{pk}'")
        if len(unmapped_pebal_keys) > 5: logger.info("  - ...")


    # --- 7. Load the updated state dict into the Hopfield model ---
    # Use strict=False. This is crucial because the Hopfield-PEBAL model is expected
    # to have additional layers (like Hopfield layers) that ARE NOT present in the
    # PEBAL checkpoint. strict=True would raise an error for these missing keys.
    try:
        missing_keys, unexpected_keys = hopfield_model.load_state_dict(hopfield_state_dict, strict=False)
    except Exception as e:
        logger.error(f"Error during final load_state_dict: {e}", exc_info=True)
        raise RuntimeError("Failed to load the modified state dict into the Hopfield model.") from e

    # Log information about the loading process results
    logger.info(f"--- load_state_dict Results (strict=False) ---")
    logger.info(f"Keys missing in loaded state_dict (exist in Hopfield model but not in PEBAL source/mapping): {len(missing_keys)}")
    # These keys correspond to layers unique to Hopfield-PEBAL (e.g., Hopfield layers) or layers intentionally not loaded.
    if missing_keys:
        for i, key in enumerate(missing_keys[:5]): logger.info(f"  - Missing Key Example {i+1}: {key}")
        if len(missing_keys) > 5: logger.info("  - ...")

    # Unexpected keys should ideally be empty if our mapping logic is perfect,
    # or correspond to the 'unmapped_pebal_keys' and 'skipped_due_to_mapping_issue'.
    logger.info(f"Unexpected keys in loaded state_dict (exist in source state_dict but not in Hopfield model): {len(unexpected_keys)}")
    if unexpected_keys:
        # This indicates keys present in the `hopfield_state_dict` *after* our transfer
        # that were *not* found in the `hopfield_model`'s structure during `load_state_dict`.
        # This situation is unusual if `hopfield_state_dict` came directly from `hopfield_model.state_dict()` initially.
        # It might point to dynamic model changes or errors in state_dict handling.
        logger.warning("Presence of unexpected keys during load_state_dict is unusual and may indicate an issue.")
        for i, key in enumerate(unexpected_keys[:5]): logger.warning(f"  - Unexpected Key Example {i+1}: {key}")
        if len(unexpected_keys) > 5: logger.warning("  - ...")
        # Cross-check with previously identified unmapped/skipped keys:
        mismatch_count = 0
        for key in unexpected_keys:
             if key not in unmapped_pebal_keys and key not in skipped_due_to_mapping_issue:
                  mismatch_count +=1
                  if mismatch_count <= 5: # Log first few discrepancies
                       logger.warning(f"    - Discrepancy: Unexpected key '{key}' was not previously identified as unmapped/skipped.")
        if mismatch_count > 5: logger.warning(f"    - ... and {mismatch_count - 5} more discrepancies.")


    logger.info("Successfully integrated transferable PEBAL weights into Hopfield-PEBAL model.")
    return hopfield_model

# --- Example Usage (requires dummy model definitions and a checkpoint file) ---
if __name__ == '__main__':

    # Configure logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Dummy Model Definitions ---
    # Simplified structures matching the key_mapping logic
    class DummyPEBALModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Structure using 'branch1' prefix
            self.branch1 = torch.nn.Module()
            self.branch1.mod1 = torch.nn.Conv2d(3, 64, 3, padding=1) # Mapped
            self.branch1.mod2 = torch.nn.ReLU()                    # Mapped
            self.branch1.pool2 = torch.nn.MaxPool2d(2)              # Mapped
            self.branch1.aspp = torch.nn.Conv2d(64, 128, 1)         # Mapped
            self.branch1.final = torch.nn.Conv2d(128, 1, 1)         # Mapped
            self.branch1.shape_mismatch_src = torch.nn.Conv2d(128, 5, 1) # Will cause shape mismatch
            self.extra_pebal_layer = torch.nn.Linear(10, 5)        # Unmapped

    class DummyHopfieldPEBALModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Structure using 'backbone' and 'segmentation_head' prefixes
            self.backbone = torch.nn.Module()
            self.backbone.mod1 = torch.nn.Conv2d(3, 64, 3, padding=1) # Matches PEBAL mod1
            self.backbone.mod2 = torch.nn.ReLU()                    # Matches PEBAL mod2
            self.backbone.pool2 = torch.nn.MaxPool2d(2)              # Matches PEBAL pool2
            self.backbone.extra_hopfield_conv = torch.nn.Conv2d(64, 64, 1) # Unique to Hopfield

            self.segmentation_head = torch.nn.Module()
            self.segmentation_head.aspp = torch.nn.Conv2d(64, 128, 1) # Matches PEBAL aspp
            self.segmentation_head.final = torch.nn.Conv2d(128, 1, 1) # Matches PEBAL final
            # Intentionally different shape for the mismatch test
            self.segmentation_head.shape_mismatch_target = torch.nn.Conv2d(128, 10, 1)

            self.hopfield_layers = torch.nn.Module() # Unique to Hopfield
            self.hopfield_layers.hop1 = torch.nn.Linear(100, 100)
            self.hopfield_layers.hop2 = torch.nn.Linear(100, 100)

    # --- Create Dummy Checkpoint ---
    pebal_model = DummyPEBALModel()
    pebal_checkpoint_path = "dummy_pebal_checkpoint.pth"
    # Save using the 'model' key structure
    torch.save({'model': pebal_model.state_dict()}, pebal_checkpoint_path)
    logger.info(f"Dummy PEBAL checkpoint created at {pebal_checkpoint_path}")

    # --- Create Hopfield Model Instance ---
    hopfield_model_instance = DummyHopfieldPEBALModel()

    # Add the specific mapping for the shape mismatch test case
    # We modify the global key_mapping used by the function for this test
    integrate_pebal_weights.__globals__['key_mapping']['branch1.shape_mismatch_src'] = 'segmentation_head.shape_mismatch_target'


    # --- Test the function ---
    # Store initial weights to verify changes
    initial_weight_mod1 = hopfield_model_instance.backbone.mod1.weight.clone().detach()
    initial_weight_hop1 = hopfield_model_instance.hopfield_layers.hop1.weight.clone().detach()

    try:
        logger.info("\n--- Starting Weight Integration ---")
        updated_model = integrate_pebal_weights(hopfield_model_instance, pebal_checkpoint_path)
        logger.info("--- Weight Integration Finished ---\n")

        # --- Verification ---
        logger.info("--- Verification Checks ---")
        # 1. Check if a mapped weight was updated
        updated_weight_mod1 = updated_model.backbone.mod1.weight
        pebal_weight_mod1 = pebal_model.branch1.mod1.weight
        assert not torch.equal(initial_weight_mod1, updated_weight_mod1), "Weight backbone.mod1 was not updated!"
        assert torch.equal(updated_weight_mod1, pebal_weight_mod1), "Weight backbone.mod1 does not match PEBAL source!"
        logger.info("✅ Verified: Weight transferred correctly for backbone.mod1.")

        # 2. Check if a Hopfield-specific layer weight remained unchanged
        updated_weight_hop1 = updated_model.hopfield_layers.hop1.weight
        assert torch.equal(initial_weight_hop1, updated_weight_hop1), "Hopfield-specific weight hopfield_layers.hop1 changed!"
        logger.info("✅ Verified: Hopfield-specific weight hopfield_layers.hop1 remains unchanged.")

        # 3. Check if the model structure is intact (Hopfield layers still exist)
        assert hasattr(updated_model, 'hopfield_layers') and hasattr(updated_model.hopfield_layers, 'hop1'), "Hopfield-specific layers missing after integration."
        logger.info("✅ Verified: Hopfield-specific model structure remains intact.")

        # 4. Check if parameters still require gradients (if they did before)
        assert updated_model.backbone.mod1.weight.requires_grad == initial_weight_mod1.requires_grad, "Gradient requirement changed for transferred weight."
        assert updated_model.hopfield_layers.hop1.weight.requires_grad == initial_weight_hop1.requires_grad, "Gradient requirement changed for Hopfield weight."
        logger.info("✅ Verified: Parameter requires_grad flags maintained.")

        logger.info("--- All Verification Checks Passed ---")

    except Exception as e:
        logger.error(f"An error occurred during the integration or verification: {e}", exc_info=True)
    finally:
        # --- Clean up dummy file ---
        if os.path.exists(pebal_checkpoint_path):
            os.remove(pebal_checkpoint_path)
            logger.info(f"Cleaned up dummy checkpoint file: {pebal_checkpoint_path}")