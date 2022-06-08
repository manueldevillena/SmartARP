from enum import Enum

import pandas as pd


class BidStatus(Enum):
    """
    Status of a bid
    """

    FREE = "free"
    REVOKED = "revoked"
    PENDING = "pending"
    EXPIRED = "expired"
    REJECTED = "rejected"
    RESERVED = "reserved"
    ACCEPTED = "accepted"


class FlexibilityBid:
    """
    Flexibility bid.
    """

    def __init__(self, flexibility: pd.Series, cost: float, owner: object, market: str,
                 expiration: pd.Timestamp = None):
        """
        Constructor.

        :param flexibility: Flexibility energy volume.
        :param cost: Cost of the flexibility bid.
        :param expiration: Expiration time of the offer.
        """

        self.flexibility = flexibility
        self.cost = cost
        self._expiration = expiration if expiration is not None else self.flexibility.index[0]
        self.acceptance = None  #: Acceptance status either None or a float in [0,1].
        self._status = BidStatus.FREE  #: Bid status.
        self.owner = owner  #: Owner of the bid.
        self.market = market

    def __lt__(self, obj2):
        return id(self) < id(obj2)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_status):
        # Check it is a valid transition
        if (
            (new_status == BidStatus.REVOKED and self._status != BidStatus.FREE)
            or (new_status == BidStatus.EXPIRED and self.status not in [BidStatus.FREE, BidStatus.RESERVED])
            or (new_status == BidStatus.PENDING and self._status != BidStatus.FREE)
            or (new_status == BidStatus.RESERVED and self._status != BidStatus.PENDING)
            or (new_status in [BidStatus.ACCEPTED, BidStatus.REJECTED] and self._status != BidStatus.RESERVED)
            or (new_status == BidStatus.FREE and self.status != BidStatus.PENDING)
        ):
            raise ValueError("Cannot make the transition {} to {}".format(self._status.value, new_status.value))

        # Apply the status
        self._status = new_status

    def check_expiration(self, now: pd.Timestamp) -> bool:
        """
        Check the expiration time of the bid and change the status if the bid expired.

        :param now: Current simulation time.
        :return: True if the bid expired
        """
        if self._expiration <= now:
            try:
                self.status = BidStatus.EXPIRED
            finally:
                return True
        return False
