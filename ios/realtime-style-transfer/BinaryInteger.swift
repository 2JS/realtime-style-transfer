extension BinaryInteger {
    func aligned(to: Self) -> Self {
        ((self - 1) / to + 1) * to
    }
}
