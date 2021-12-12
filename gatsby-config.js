/**
 * Configure your Gatsby site with this file.
 *
 * See: https://www.gatsbyjs.com/docs/gatsby-config/
 */

module.exports = {
	/* Your site config here */
	plugins: [
		"gatsby-transformer-sharp",
		"gatsby-plugin-sharp",
		// {
		// 	resolve: `gatsby-source-filesystem`,
		// 	options: {
		// 		name: `images`,
		// 		path: `${__dirname}/src/images/`,
		// 	},
		// },
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				path: `${__dirname}/static/assets/images/uploads`,
				name: "uploads",
			},
		},
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				name: `posts`,
				path: `${__dirname}/src/posts/`,
			},
		},
		{
			resolve: "gatsby-transformer-remark",
			options: {
				plugins: [
					{
						resolve: "gatsby-remark-relative-images",
						options: {
							name: "uploads", // Must match the source name ^
						},
					},
					{
						resolve: `gatsby-remark-images`,
						options: {
							maxWidth: 590,
						},
					},
				],
			},
		},
		"gatsby-plugin-netlify-cms",
	],
	siteMetadata: {
		title: "The Codacus",
		description: "The Coding Abacus",
		copyright: "Copyright 2021 The Codacus",
	},
}
