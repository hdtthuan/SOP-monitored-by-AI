class SOPMonitoring:
    """
    Monitors a sequence of actions and ensures they follow the predefined order.

    For each step, the current expected action is "held" until a new action (the next expected one)
    is detected. In the initial step, the two actions ("get [Clamp Bracket]" and "get [L-Shaped Bracket]")
    can occur in any order. The system remains at the initial step until both have been detected and then
    transitions only when a new action (e.g., "get screw [1]") is detected.
    """

    def __init__(self):
        # Define the required SOP sequence.
        # Step 0 is a set of interchangeable actions.
        self.sop_sequence = [
            {"get [Clamp Bracket]", "get [L-Shaped Bracket]"},  # Step 0: interchangeable initial actions
            "get screw [1]",                                    # Step 1
            "get [Allen Key]",                                  # Step 2
            "Put [Allen Key]",                                  # Step 3
            "get [Sliding Connector]",                          # Step 4
            "get screw [2]",                                    # Step 5
            "get [Allen Key]",                                  # Step 6
            "Put [Allen Key]",                                  # Step 7
            "get [Connector Block]",                            # Step 8
            "get screw [3]",                                    # Step 9
            "get [Allen Key]",                                  # Step 10
            "Put [Allen Key]",                                  # Step 11
        ]
        self.current_step = 0                     # Index in the SOP sequence
        self.completed_initial_actions = set()    # Track which actions in step 0 have been detected
        self.waiting_for_initial_actions = True   # True while still in step 0
        self.last_validated_action = None         # Holds the action that confirmed the current step

    def validate_action(self, action):
        """
        Validates the detected action against the expected SOP sequence.

        - For the initial step, if action is in the expected set, it is recorded.
          Once both initial actions are detected, the system remains at step 0 until a new action is detected.
        - For subsequent steps, if the same action is detected continuously, the system "holds" the step.
          Only when a new action (matching the next expected action) is detected, the step is advanced.

        Parameters:
          - action (str): The detected action (e.g., "get [Clamp Bracket]")

        Returns:
          - True if the action is valid (either holding the current step or transitioning properly).
          - False if the action is out of order.
        """
        # If all steps have been completed
        if self.current_step >= len(self.sop_sequence):
            print("[SOP Completed] All required actions have been performed.")
            return True

        # --- process step 0 (interchangeable) ---
        if self.waiting_for_initial_actions:
            expected_set = self.sop_sequence[0]
            if action in expected_set:
                # inital the first step
                if action not in self.completed_initial_actions:
                    self.completed_initial_actions.add(action)
                    print(f"[VALID] Completed initial action: {action}. Waiting for the other.")
                else:
                    print(f"[INFO] Initial action already recorded: {action}.")
                # waiting for both actions in step 0 be completed
                print(f"[INFO] Currently completed: {self.completed_initial_actions}")
                return True
            else:
                # if step 0 is completed, moving to the next step
                if self.completed_initial_actions == expected_set:
                    next_expected = self.sop_sequence[1]
                    if action == next_expected:
                        self.current_step = 1
                        self.waiting_for_initial_actions = False
                        self.last_validated_action = action
                        print(f"[VALID] Transitioning to next action: {action}")
                        return True
                    else:
                        print(f"[ERROR] Incorrect action: {action}. Expected: {next_expected}")
                        return False
                else:
                    print(f"[ERROR] Incorrect action: {action}. Expected one of initial actions: {expected_set}")
                    return False

        # --- process next steps ---
        expected = self.sop_sequence[self.current_step]
        # Waiting for the next step and hold it in the current step
        if self.last_validated_action is None:
            if action == expected:
                self.last_validated_action = action
                print(f"[VALID] Holding current action: {action}.")
                return True
            else:
                print(f"[ERROR] Incorrect action: {action}. Expected: {expected}")
                return False
        else:
            # If detected action is still the same with the current actions, stay in that step
            if action == self.last_validated_action:
                print(f"[INFO] Continuing to hold current action: {action}.")
                return True
            else:
                # The next action should fit the next step
                if self.current_step + 1 < len(self.sop_sequence):
                    next_expected = self.sop_sequence[self.current_step + 1]
                    if action == next_expected:
                        self.current_step += 1
                        self.last_validated_action = action
                        print(f"[VALID] Transitioning to next action: {action}")
                        return True
                    else:
                        print(f"[ERROR] Incorrect action: {action}. Expected next action: {next_expected}")
                        return False
                else:
                    print(f"[ERROR] No next action expected, but received: {action}")
                    return False
    def reset_monitoring(self):
        """Reset SOP sequence to allow re-monitoring from the beginning."""
        self.current_step = 0
        self.completed_initial_actions.clear()
        self.last_validated_action = None
        self.waiting_for_initial_actions = True
        print("[INFO] SOP monitoring reset.")
    def get_expected_action(self):
        # Return the current expected action from SOP
        if self.current_step < len(self.sop_sequence):
            return self.sop_sequence[self.current_step]
        return None


    def is_sop_complete(self):
        """
        Checks if all SOP actions have been successfully completed.

        Returns:
          - True if the SOP sequence is completed.
          - False otherwise.
        """
        return self.current_step >= len(self.sop_sequence)
    