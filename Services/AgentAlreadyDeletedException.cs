using System;

namespace TransferLearning.Services
{
    /// <summary>
    /// Exception raised when a delete operation is attempted on an agent that is already
    /// in deleted status. This is a business-rule violation and should be presented to
    /// the caller as a user-friendly error rather than a system error.
    /// </summary>
    public class AgentAlreadyDeletedException : InvalidOperationException
    {
        public AgentAlreadyDeletedException(string agentID)
            : base($"Bu bayi zaten silinmiş durumda. İşlem gerçekleştirilemez. AgentID: {agentID}")
        {
        }
    }
}
