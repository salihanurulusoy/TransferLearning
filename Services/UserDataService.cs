using System;
using TransferLearning.Data;
using TransferLearning.Models;

namespace TransferLearning.Services
{
    public enum AgentSyncType
    {
        AddAgent = 1,
        UpdateAgentActiveStatus = 2
    }

    public class UserDataService
    {
        private readonly string companyCode;

        public UserDataService(string companyCode)
        {
            this.companyCode = companyCode;
        }

        /// <summary>
        /// Retrieves an agent directly from the database by ID.
        /// </summary>
        public Agent GetAgentByIDFromDb(string agentID)
        {
            // Implementation fetches agent from database.
            throw new NotImplementedException();
        }

        /// <summary>
        /// Retrieves an agent from the in-memory cache by ID.
        /// </summary>
        public Agent GetPaynetAgentFromCache(string agentID)
        {
            // Implementation fetches agent from cache.
            throw new NotImplementedException();
        }

        /// <summary>
        /// Deletes the specified agent.
        /// Throws <see cref="InvalidOperationException"/> if the agent is already in deleted status,
        /// preventing a duplicate note from being inserted and a no-op delete from being executed.
        /// </summary>
        /// <param name="agentID">The ID of the agent to delete.</param>
        /// <param name="note">An optional reason/note for the deletion.</param>
        /// <param name="disableSanctionScan">When true, the post-delete sanction scan is skipped.</param>
        /// <returns>True if the agent was successfully deleted.</returns>
        /// <exception cref="AgentAlreadyDeletedException">
        /// Thrown when the agent is already in deleted status.
        /// </exception>
        public bool AgentDelete(string agentID, string note, bool disableSanctionScan = false)
        {
            int xactNoteID = 0;

            // Guard: do not allow re-deletion of an already-deleted agent.
            // This prevents an orphaned note from being written and avoids a
            // misleading success result when the agent's status would not change.
            var existingAgent = GetPaynetAgentFromCache(agentID);
            if (existingAgent != null && existingAgent.IsAgentDeleted)
                throw new AgentAlreadyDeletedException(agentID);

            if (!string.IsNullOrWhiteSpace(note))
                xactNoteID = UserDatabase.InsertNote(note);

            bool result = UserDatabase.AgentDelete(this.companyCode, agentID, xactNoteID);
            if (result)
            {
                UpdateIsSentToCrmAsFalseForAgent(agentID);
                UpdateIsSentToBkmAsFalseForAgent(companyCode, agentID);
                var agent = GetPaynetAgentFromCache(agentID);
                if (agent != null)
                {
                    int syncType = string.IsNullOrEmpty(agent.IyzicoMerchantId)
                        ? (int)AgentSyncType.AddAgent
                        : (int)AgentSyncType.UpdateAgentActiveStatus;
                    SendSyncIyzicoMerchantCommand(agentID, syncType, string.Empty);
                }
                if (!disableSanctionScan)
                    ProcessSanctionScan(agentID);
            }

            RefreshGetAllPaynetAgent(agentID);

            return result;
        }

        private void UpdateIsSentToCrmAsFalseForAgent(string agentID)
        {
            throw new NotImplementedException();
        }

        private void UpdateIsSentToBkmAsFalseForAgent(string companyCode, string agentID)
        {
            throw new NotImplementedException();
        }

        private void SendSyncIyzicoMerchantCommand(string agentID, int syncType, string extra)
        {
            throw new NotImplementedException();
        }

        private void ProcessSanctionScan(string agentID)
        {
            throw new NotImplementedException();
        }

        private void RefreshGetAllPaynetAgent(string agentID)
        {
            throw new NotImplementedException();
        }
    }
}
